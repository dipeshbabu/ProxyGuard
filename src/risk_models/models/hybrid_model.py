"""Hybrid quantum-classical risk model."""

import shap
from xgboost import XGBClassifier

from risk_models.config import HYPERPARAMETERS
from risk_models.feature_engineering.selectors import QuantumFeatureSelector
from risk_models.feature_engineering.segmenter import QuantumBusinessSegmenter


class HybridRiskModel:
    """Hybrid risk model combining quantum-inspired features with classical ML."""
    
    def __init__(self, use_segmentation=True, use_feature_selection=True):
        """Initialize the hybrid risk model.
        
        Args:
            use_segmentation: Whether to use quantum business segmentation
            use_feature_selection: Whether to use quantum feature selection
        """
        self.use_segmentation = use_segmentation
        self.use_feature_selection = use_feature_selection
        self.feature_selector = QuantumFeatureSelector()
        self.classifier = XGBClassifier(**HYPERPARAMETERS['xgb'])
        self.segmenter = QuantumBusinessSegmenter()
        self.selected_features = None
        self.explainer = None

    def fit(self, X, y):
        """Fit the model to the training data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Self
        """
        X_segmented = X.copy()
        if self.use_segmentation:
            try:
                self.segmenter.fit(X_segmented)
                X_segmented['Segment'] = self.segmenter.predict(X_segmented)
            except:
                X_segmented['Segment'] = 0

        if self.use_feature_selection:
            try:
                self.selected_features = self.feature_selector.optimize(
                    X_segmented, y)
            except:
                self.selected_features = X_segmented.columns[:10]

        if len(self.selected_features) == 0:
            self.selected_features = X_segmented.columns[:10]

        X_selected = X_segmented[self.selected_features]
        self.classifier.fit(X_selected, y)
        self.explainer = shap.TreeExplainer(self.classifier)
        return self

    def predict(self, X):
        """Predict risk labels for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted risk labels
        """
        X_segmented = X.copy()
        if self.use_segmentation:
            try:
                X_segmented['Segment'] = self.segmenter.predict(X_segmented)
            except:
                X_segmented['Segment'] = 0
        return self.classifier.predict(X_segmented[self.selected_features])
        
    def predict_proba(self, X):
        """Predict risk probabilities for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted risk probabilities
        """
        X_segmented = X.copy()
        if self.use_segmentation:
            try:
                X_segmented['Segment'] = self.segmenter.predict(X_segmented)
            except:
                X_segmented['Segment'] = 0
        return self.classifier.predict_proba(X_segmented[self.selected_features]) 