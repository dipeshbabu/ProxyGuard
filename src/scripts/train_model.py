#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to train and save a hybrid risk model."""

import argparse
import time
import sys
import os
import logging

# Add src directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from risk_models.data import preprocess_data
from risk_models.models import HybridRiskModel
from risk_models.utils.io import save_model
from risk_models.evaluation.metrics import analyze_performance, interpret_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a hybrid risk model.')
    parser.add_argument('--use-segmentation', action='store_true', 
                        help='Use quantum-inspired business segmentation')
    parser.add_argument('--use-feature-selection', action='store_true',
                        help='Use quantum-inspired feature selection')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to save the trained model')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information during training')
    
    return parser.parse_args()


def main():
    """Train and evaluate the hybrid risk model."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data()
    logger.info(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Initialize model
    logger.info("Initializing model...")
    model = HybridRiskModel(
        use_segmentation=args.use_segmentation,
        use_feature_selection=args.use_feature_selection
    )
    
    # Train model
    logger.info("Training model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    logger.info(f"Model trained in {training_time:.2f} seconds")
    
    # Evaluate model
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    metrics = analyze_performance(y_test, y_pred, "Hybrid Risk Model")
    
    # Print results
    if args.verbose:
        interpret_results(metrics, "Hybrid Risk Model")
        print(f"Training time: {training_time:.2f} seconds")
        
        if model.use_feature_selection and model.selected_features is not None:
            print(f"Selected features: {', '.join(model.selected_features)}")
    
    # Save model
    logger.info("Saving model...")
    saved_path = save_model(model, args.output_path)
    logger.info(f"Model saved to {saved_path}")
    
    return metrics


if __name__ == "__main__":
    main() 