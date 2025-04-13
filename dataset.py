# -*- coding: utf-8 -*-
# dataset.py - Data loading and preprocessing

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.combine import SMOTEENN
from configs import (DATASET_PATH, CATEGORICAL_FEATURES,
                     RANDOM_STATE, TEST_SIZE, SPLIT_RANDOM_STATE)


def adapt_to_small_business(df):
    df['Business_Age'] = df['Age'].apply(lambda x: max(0, x - 22))
    df['Employees'] = df['Job'].map({
        0: '1-5', 1: '1-5', 2: '5-10', 3: '10-20'
    }).astype('category')
    df['Monthly_Revenue'] = df['Credit amount'] / df['Duration'].replace(0, 1)
    df['Business_Type'] = pd.cut(
        df['Credit amount'],
        bins=[0, 5000, 20000, np.inf],
        labels=['Micro', 'Small', 'Medium']
    )
    return df


def create_risk_label(df):
    df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
    df['Checking account'] = df['Checking account'].fillna('unknown')

    savings_order = {'unknown': 0, 'little': 1,
                     'moderate': 2, 'quite rich': 3, 'rich': 4}
    checking_order = {'unknown': 0, 'little': 1, 'moderate': 2, 'rich': 3}

    df['credit_ratio'] = df['Credit amount'] / \
        df['Duration'].replace(0, np.nan)
    df['credit_ratio'] = df['credit_ratio'].fillna(df['credit_ratio'].mean())
    df['account_mismatch'] = abs(
        df['Saving accounts'].map(savings_order) -
        df['Checking account'].map(checking_order)
    )

    risk_conditions = (
        (df['credit_ratio'] > 500) |
        (df['account_mismatch'] > 2) |
        (df['Job'] < 1) |
        (df['Purpose'].isin(['radio/TV', 'education']))
    )
    df['Risk'] = np.where(risk_conditions, 1, 0)
    return df.drop(['credit_ratio', 'account_mismatch'], axis=1)


def preprocess_data():
    df = pd.read_csv(DATASET_PATH, index_col=0)
    df = adapt_to_small_business(df)
    df = create_risk_label(df)

    # Encoding
    for col, cats in CATEGORICAL_FEATURES.items():
        if col in ['Saving accounts', 'Checking account']:
            encoder = OrdinalEncoder(
                categories=[cats], handle_unknown='use_encoded_value', unknown_value=-1)
            df[col] = encoder.fit_transform(df[[col]])

    df = pd.get_dummies(df, columns=['Sex', 'Housing', 'Purpose', 'Employees', 'Business_Type'],
                        drop_first=False, dummy_na=True)

    # Numeric features
    numeric_features = ['Business_Age',
                        'Credit amount', 'Duration', 'Monthly_Revenue']
    df[numeric_features] = StandardScaler().fit_transform(df[numeric_features])
    df = df.fillna(df.mean())

    # Balance classes
    X = df.drop('Risk', axis=1)
    y = df['Risk']
    X_res, y_res = SMOTEENN(random_state=RANDOM_STATE).fit_resample(X, y)

    return train_test_split(
        X_res, y_res,
        test_size=TEST_SIZE,
        stratify=y_res,
        random_state=SPLIT_RANDOM_STATE
    )
