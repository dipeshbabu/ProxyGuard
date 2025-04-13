"""Visualization utilities for risk models."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_feature_importances(model, feature_names=None):
    """Plot feature importances from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
    """
    if not hasattr(model.classifier, 'feature_importances_'):
        raise AttributeError("Model does not have feature_importances_ attribute")
        
    importances = model.classifier.feature_importances_
    
    if feature_names is None:
        if model.selected_features is not None:
            feature_names = model.selected_features
        else:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    
def plot_segment_distribution(X, segmenter, feature1='Credit amount', feature2='Monthly_Revenue'):
    """Plot the distribution of business segments.
    
    Args:
        X: Feature matrix
        segmenter: Trained segmenter model
        feature1: First feature to plot
        feature2: Second feature to plot
    """
    segments = segmenter.predict(X)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=X[feature1],
        y=X[feature2],
        hue=segments,
        palette='viridis',
        s=100,
        alpha=0.7
    )
    plt.title('Business Segments Distribution')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend(title='Segment')
    
    # Add segment centers if available
    if hasattr(segmenter, 'kmeans') and hasattr(segmenter.kmeans, 'cluster_centers_'):
        # We need to transform centers back to original scale
        financial_features = X[[feature1, feature2]].columns
        all_financial_features = X[['Credit amount', 'Duration', 'Monthly_Revenue']]
        
        # Get indices of the features we want to plot
        idx1 = list(all_financial_features.columns).index(feature1)
        idx2 = list(all_financial_features.columns).index(feature2)
        
        # Transform cluster centers back
        scaled_centers = segmenter.kmeans.cluster_centers_
        centers = segmenter.scaler.inverse_transform(scaled_centers)
        
        plt.scatter(
            centers[:, idx1],
            centers[:, idx2],
            s=200,
            c='red',
            marker='X',
            label='Segment Centers'
        )
        
    plt.show()
    
    
def visualize_data_distribution(X, y=None, method='pca'):
    """Visualize data distribution using dimensionality reduction.
    
    Args:
        X: Feature matrix
        y: Optional target vector for coloring
        method: Dimensionality reduction method: 'pca' or 'tsne'
    """
    if method.lower() == 'pca':
        model = PCA(n_components=2)
    elif method.lower() == 'tsne':
        model = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
        
    X_reduced = model.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    
    if y is not None:
        sns.scatterplot(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1],
            hue=y,
            palette='viridis',
            s=100,
            alpha=0.7
        )
        plt.title(f'Data Distribution ({method.upper()}) by Target')
        plt.legend(title='Risk')
    else:
        sns.scatterplot(
            x=X_reduced[:, 0],
            y=X_reduced[:, 1],
            s=100,
            alpha=0.7
        )
        plt.title(f'Data Distribution ({method.upper()})')
        
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show() 