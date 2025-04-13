"""Business segmentation components."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from neal import SimulatedAnnealingSampler

from risk_models.config import N_CLUSTERS, RANDOM_STATE, HYPERPARAMETERS


class QuantumBusinessSegmenter:
    """Business segmenter using quantum-inspired algorithms."""
    
    def __init__(self, n_clusters=N_CLUSTERS):
        """Initialize the business segmenter.
        
        Args:
            n_clusters: Number of business segments to create
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        self.init_centers = None

    def fit(self, X):
        """Fit the segmenter to the data.
        
        Args:
            X: Feature matrix with financial columns
            
        Returns:
            Self
        """
        financial_features = X[['Credit amount',
                                'Duration', 'Monthly_Revenue']]
        scaled_features = self.scaler.fit_transform(financial_features)

        n_samples = scaled_features.shape[0]
        qubo = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                qubo[i, j] = -np.dot(scaled_features[i], scaled_features[j])

        response = SimulatedAnnealingSampler().sample_qubo(
            qubo,
            num_reads=HYPERPARAMETERS['sampler']['segmenter_reads']
        )
        centers_idx = list(response.first.sample.keys())[:self.n_clusters]
        self.init_centers = scaled_features[centers_idx]

        self.kmeans.init = self.init_centers
        self.kmeans.fit(scaled_features)
        return self

    def predict(self, X):
        """Predict segments for new data.
        
        Args:
            X: Feature matrix with financial columns
            
        Returns:
            Segment labels
        """
        financial_features = X[['Credit amount',
                                'Duration', 'Monthly_Revenue']]
        scaled_features = self.scaler.transform(financial_features)
        return self.kmeans.predict(scaled_features) 