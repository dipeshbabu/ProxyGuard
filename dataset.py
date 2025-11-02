# dataset.py (leakage-safe derived label)
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Columns that define the derived label or are direct transforms of them
LEAK_RAW = [
    'Credit amount', 'Duration',           # used in credit_ratio
    'Monthly_Revenue',                     # = Credit amount / Duration
    'Job', 'Employees',                    # Employees derived from Job
    'Business_Type',                       # derived from Credit amount
    'Saving accounts', 'Checking account',  # used directly in rule
    'Purpose'                              # used directly in rule (before OHE)
]
LEAK_PREFIXES = [
    'Purpose_',        # OHE of Purpose
    'Employees_',      # OHE of Employees
    'Business_Type_'   # OHE of Business_Type
]


def adapt_to_small_business(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Business_Age'] = df['Age'].apply(lambda x: max(0, x - 22))
    df['Employees'] = df['Job'].map(
        {0: '1-5', 1: '1-5', 2: '5-10', 3: '10-20'}).astype('category')
    df['Monthly_Revenue'] = df['Credit amount'] / df['Duration'].replace(0, 1)
    df['Business_Type'] = pd.cut(
        df['Credit amount'],
        bins=[0, 5000, 20000, np.inf],
        labels=['Micro', 'Small', 'Medium']
    )
    return df


def create_risk_label(df: pd.DataFrame, top_frac: float = 0.35) -> pd.DataFrame:
    """
    Derive Risk from the upper quantile of credit_ratio only (no accounts/purpose/job in rule).
    This avoids dropping too many useful predictors later.
    """
    df = df.copy()
    df['credit_ratio'] = df['Credit amount'] / df['Duration'].replace(0, 1)
    thr = df['credit_ratio'].quantile(1.0 - top_frac)  # e.g., top 30% = risky
    df['Risk'] = (df['credit_ratio'] >= thr).astype(int)
    return df.drop(columns=['credit_ratio'])


# Features to drop to prevent leakage (parents of the label and their direct transforms)
LEAK_RAW = [
    # parents / direct function of parents
    'Credit amount', 'Duration', 'Monthly_Revenue'
]
LEAK_PREFIXES = [
    'Business_Type_'  # derived deterministically from Credit amount
]


def preprocess_data(filepath: str):
    df = pd.read_csv(filepath, index_col=0)

    # business adds (OK to keep; we will drop the leaking ones below)
    df['Business_Age'] = df['Age'].apply(lambda x: max(0, x - 22))
    df['Employees'] = df['Job'].map(
        {0: '1-5', 1: '1-5', 2: '5-10', 3: '10-20'}).astype('category')
    df['Monthly_Revenue'] = df['Credit amount'] / df['Duration'].replace(0, 1)
    df['Business_Type'] = pd.cut(df['Credit amount'], bins=[0, 5000, 20000, np.inf],
                                 labels=['Micro', 'Small', 'Medium'])

    # derive label only from credit_ratio quantile
    df = create_risk_label(df, top_frac=0.30)

    # minimal cleaning/encoding (DO NOT use these in the label)
    df['Saving accounts'] = df['Saving accounts'].fillna('NaN')
    df['Checking account'] = df['Checking account'].fillna('NaN')
    cat_spec = {
        'Saving accounts': ['NaN', 'little', 'moderate', 'quite rich', 'rich'],
        'Checking account': ['NaN', 'little', 'moderate', 'rich'],
    }
    from sklearn.preprocessing import OrdinalEncoder
    for col in ['Saving accounts', 'Checking account']:
        enc = OrdinalEncoder(categories=[cat_spec[col]],
                             handle_unknown='use_encoded_value', unknown_value=np.nan)
        df[col] = enc.fit_transform(df[[col]])

    df = pd.get_dummies(df,
                        columns=['Sex', 'Housing', 'Purpose',
                                 'Employees', 'Business_Type'],
                        dummy_na=True, drop_first=False)

    leak_parents = [
        'Credit amount', 'Duration',
        'Saving accounts', 'Checking account',
    ]
    leak_parents_oh = [c for c in df.columns if c.startswith('Purpose_')]
    df = df.drop(columns=[c for c in leak_parents +
                          leak_parents_oh if c in df.columns], errors='ignore')

    # df = df.dropna()

    # Build X then drop leakage columns
    X = df.drop('Risk', axis=1)
    to_drop = [c for c in LEAK_RAW if c in X.columns]
    to_drop += [c for c in X.columns if any(c.startswith(p)
                                            for p in LEAK_PREFIXES)]
    X = X.drop(columns=to_drop, errors='ignore')

    y = df['Risk'].astype(int)

    # Numeric columns left (for scalers, etc.)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    return X, y, numeric_cols
