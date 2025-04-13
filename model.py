# -*- coding: utf-8 -*-
# model.py - Model components

import numpy as np
from sklearn.preprocessing import StandardScaler
from neal import SimulatedAnnealingSampler
import shap
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
import shap
from configs import (NUM_FEATURES, N_CLUSTERS, HYPERPARAMETERS, RANDOM_STATE)


class QuantumFeatureSelector:
    def __init__(self, num_features=NUM_FEATURES):
        self.num_features = num_features
        self.sampler = SimulatedAnnealingSampler()
        self.selected_features = None

    def create_qubo(self, X, y):
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


class QuantumBusinessSegmenter:
    def __init__(self, n_clusters=N_CLUSTERS):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
        self.init_centers = None

    def fit(self, X):
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
        financial_features = X[['Credit amount',
                                'Duration', 'Monthly_Revenue']]
        scaled_features = self.scaler.transform(financial_features)
        return self.kmeans.predict(scaled_features)


class HybridRiskModel:
    def __init__(self, use_segmentation=True, use_feature_selection=True):
        self.use_segmentation = use_segmentation
        self.use_feature_selection = use_feature_selection
        self.feature_selector = QuantumFeatureSelector()
        self.classifier = XGBClassifier(**HYPERPARAMETERS['xgb'])
        self.segmenter = QuantumBusinessSegmenter()
        self.selected_features = None
        self.explainer = None

    def fit(self, X, y):
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
        X_segmented = X.copy()
        if self.use_segmentation:
            try:
                X_segmented['Segment'] = self.segmenter.predict(X_segmented)
            except:
                X_segmented['Segment'] = 0
        return self.classifier.predict(X_segmented[self.selected_features])
