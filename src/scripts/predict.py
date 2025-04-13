#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to make predictions with a trained model."""

import argparse
import sys
import os
import logging
import pandas as pd

# Add src directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from risk_models.data import adapt_to_small_business
from risk_models.utils.io import load_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with a trained risk model.')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the trained model')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the input CSV file with data')
    parser.add_argument('--output-file', type=str, default='predictions.csv',
                        help='Path to save the predictions')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information during prediction')
    
    return parser.parse_args()


def preprocess_input(input_file):
    """Preprocess input data for prediction.
    
    Args:
        input_file: Path to input CSV file
        
    Returns:
        Preprocessed dataframe
    """
    df = pd.read_csv(input_file, index_col=0)
    df = adapt_to_small_business(df)
    
    # Ensure required columns
    required_columns = ['Credit amount', 'Duration', 'Monthly_Revenue']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Input data missing required columns: {', '.join(missing_columns)}")
    
    return df


def main():
    """Make predictions with a trained model."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load model
    logger.info("Loading model...")
    model = load_model(args.model_path)
    logger.info("Model loaded successfully")
    
    # Preprocess input data
    logger.info("Preprocessing input data...")
    X = preprocess_input(args.input_file)
    logger.info(f"Loaded {len(X)} records for prediction")
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    
    # Create output dataframe
    result = X.copy()
    result['Risk_Prediction'] = y_pred
    result['Risk_Probability'] = probas
    
    # Save results
    logger.info(f"Saving predictions to {args.output_file}...")
    result.to_csv(args.output_file)
    logger.info("Predictions saved successfully")
    
    # Print summary if verbose
    if args.verbose:
        print(f"Prediction summary:")
        print(f"Total records: {len(result)}")
        print(f"High risk cases: {sum(y_pred)} ({sum(y_pred)/len(y_pred)*100:.1f}%)")
        print(f"Low risk cases: {len(y_pred) - sum(y_pred)} ({(1-sum(y_pred)/len(y_pred))*100:.1f}%)")
    
    return result


if __name__ == "__main__":
    main() 