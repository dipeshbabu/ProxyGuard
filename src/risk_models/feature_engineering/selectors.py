"""Feature selection components."""

import numpy as np
from neal import SimulatedAnnealingSampler
from sklearn.feature_selection import mutual_info_classif

from risk_models.config import NUM_FEATURES, HYPERPARAMETERS


class QuantumFeatureSelector:
    """Feature selector using quantum-inspired algorithms."""
    
    def __init__(self, num_features=NUM_FEATURES):
        """Initialize the feature selector.
        
        Args:
            num_features: Number of features to select
        """
        self.num_features = num_features
        self.sampler = SimulatedAnnealingSampler()
        self.selected_features = None

    def create_qubo(self, X, y):
        """Create QUBO problem for feature selection.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            QUBO matrix
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        mi = mutual_info_classif(X, y)
        n_features = X.shape[1]

        try:
            corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
        except:
            corr_matrix = np.zeros((n_features, n_features))

        qubo = np.diag(-1.0 * mi) + 0.1 * \
            (np.ones_like(corr_matrix) - corr_matrix)
        np.fill_diagonal(qubo, qubo.diagonal() * 1.5)
        return qubo

    def optimize(self, X, y):
        """Optimize feature selection using simulated annealing.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Selected feature names
        """
        X_array = np.asarray(X)
        qubo = self.create_qubo(X_array, y)
        response = self.sampler.sample_qubo(
            qubo,
            num_reads=HYPERPARAMETERS['sampler']['feature_selector_reads']
        )
        selected = np.array(response.record.sample[0]).astype(bool)

        if sum(selected) == 0:
            selected[:min(self.num_features, len(selected))] = True
        elif sum(selected) < self.num_features:
            selected[np.argsort(selected)[-self.num_features:]] = True

        self.selected_features = X.columns[selected][:self.num_features]
        return self.selected_features 