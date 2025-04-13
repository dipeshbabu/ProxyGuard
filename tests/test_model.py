"""Tests for the HybridRiskModel class."""

import unittest
import numpy as np
import pandas as pd
from src.risk_models.models import HybridRiskModel


class TestHybridRiskModel(unittest.TestCase):
    """Test cases for the HybridRiskModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small synthetic dataset
        np.random.seed(42)
        n_samples = 50
        
        # Create features
        self.X = pd.DataFrame({
            'Credit amount': np.random.normal(10000, 5000, n_samples),
            'Duration': np.random.randint(12, 60, n_samples),
            'Monthly_Revenue': np.random.normal(2000, 1000, n_samples),
            'Age': np.random.randint(25, 60, n_samples),
            'Job': np.random.randint(0, 4, n_samples),
            'Saving accounts': np.random.choice(['little', 'moderate', 'rich'], n_samples),
            'Checking account': np.random.choice(['little', 'moderate', 'rich'], n_samples),
            'Sex_male': np.random.randint(0, 2, n_samples),
            'Sex_female': np.random.randint(0, 2, n_samples),
            'Housing_own': np.random.randint(0, 2, n_samples),
            'Housing_rent': np.random.randint(0, 2, n_samples),
            'Purpose_car': np.random.randint(0, 2, n_samples),
            'Purpose_business': np.random.randint(0, 2, n_samples)
        })
        
        # Create target
        self.y = np.random.randint(0, 2, n_samples)
        
        # Create model instance
        self.model = HybridRiskModel(use_segmentation=True, use_feature_selection=True)
        
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertTrue(self.model.use_segmentation)
        self.assertTrue(self.model.use_feature_selection)
        self.assertIsNotNone(self.model.feature_selector)
        self.assertIsNotNone(self.model.segmenter)
        self.assertIsNotNone(self.model.classifier)
    
    def test_model_fit_predict(self):
        """Test model fitting and prediction."""
        try:
            # Fit the model
            self.model.fit(self.X, self.y)
            
            # Check if features were selected
            self.assertIsNotNone(self.model.selected_features)
            self.assertGreater(len(self.model.selected_features), 0)
            
            # Make predictions
            y_pred = self.model.predict(self.X)
            
            # Check predictions shape and type
            self.assertEqual(len(y_pred), len(self.y))
            self.assertTrue(np.issubdtype(y_pred.dtype, np.integer))
            
            # Check that predictions are binary
            self.assertTrue(np.all(np.isin(y_pred, [0, 1])))
            
        except Exception as e:
            self.fail(f"Model fitting or prediction failed with error: {str(e)}")
    
    def test_predict_proba(self):
        """Test probability predictions."""
        # Fit the model
        self.model.fit(self.X, self.y)
        
        # Get probability predictions
        probas = self.model.predict_proba(self.X)
        
        # Check shape and values
        self.assertEqual(probas.shape, (len(self.y), 2))
        self.assertTrue(np.all(probas >= 0))
        self.assertTrue(np.all(probas <= 1))
        
        # Check that probabilities sum to 1
        self.assertTrue(np.allclose(np.sum(probas, axis=1), 1.0))


if __name__ == '__main__':
    unittest.main() 