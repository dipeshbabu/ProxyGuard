# -*- coding: utf-8 -*-
"""Central configuration hub for risk models."""

import os
from pathlib import Path

# Root directory 
ROOT_DIR = Path(__file__).parent.parent.parent

# Path configurations
DATASET_PATH = os.path.join(ROOT_DIR, 'data', 'german_credit.csv')
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'hybrid_credit_model.pkl')

# Feature engineering
NUM_FEATURES = 8
N_CLUSTERS = 3
CATEGORICAL_FEATURES = {
    'Saving accounts': ['unknown', 'little', 'moderate', 'quite rich', 'rich'],
    'Checking account': ['unknown', 'little', 'moderate', 'rich'],
    'Sex': ['male', 'female'],
    'Housing': ['own', 'rent', 'free'],
    'Purpose': ['car', 'furniture/equipment', 'radio/TV',
                'domestic appliances', 'repairs', 'education',
                'business', 'vacation/others']
}

# Model parameters
HYPERPARAMETERS = {
    'xgb': {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    },
    'sampler': {
        'feature_selector_reads': 1000,
        'segmenter_reads': 100
    }
}

# Experiment settings
RANDOM_STATE = 42
SPLIT_RANDOM_STATE = 43
TEST_SIZE = 0.2
N_BOOTSTRAP_ITERATIONS = 1000 