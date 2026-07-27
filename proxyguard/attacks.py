from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log, pi

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class AttackRepresentation:
    imputer: SimpleImputer
    scaler: StandardScaler
    pca: PCA

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        imputed = self.imputer.transform(frame)
        scaled = self.scaler.transform(imputed)
        return self.pca.transform(scaled)


def fit_attack_representation(
    reference: pd.DataFrame,
    pca_dimensions: int = 10,
) -> AttackRepresentation:
    if reference.empty:
        raise ValueError("The reference table must contain records.")
    if pca_dimensions < 1:
        raise ValueError("pca_dimensions must be positive.")
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(reference)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed)
    dimensions = min(pca_dimensions, scaled.shape[1], max(1, scaled.shape[0] - 1))
    pca = PCA(n_components=dimensions, random_state=0)
    pca.fit(scaled)
    return AttackRepresentation(imputer=imputer, scaler=scaler, pca=pca)


def domias_kde_scores(
    query: np.ndarray,
    synthetic: np.ndarray,
    reference: np.ndarray,
    seed: int,
    jitter: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the DOMIAS density-ratio and density-only log scores."""

    rng = np.random.default_rng(seed)
    synthetic_jittered = synthetic + rng.normal(0.0, jitter, synthetic.shape)
    reference_jittered = reference + rng.normal(0.0, jitter, reference.shape)
    synthetic_density = stats.gaussian_kde(synthetic_jittered.T)
    reference_density = stats.gaussian_kde(reference_jittered.T)
    log_synthetic = synthetic_density.logpdf(query.T)
    log_reference = reference_density.logpdf(query.T)
    return log_synthetic - log_reference, log_synthetic


def dcr_scores(query: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=1, metric="euclidean")
    model.fit(synthetic)
    distance, _ = model.kneighbors(query)
    return -distance[:, 0]


def gen_lra_scores(
    query: np.ndarray,
    synthetic: np.ndarray,
    reference: np.ndarray,
    neighbors: int = 10,
) -> np.ndarray:
    """Compute the KDE likelihood-influence score used by Gen-LRA.

    The score is the change in the local synthetic log likelihood after a
    query record is added to the reference KDE. The local region is fixed to
    the registered number of nearest synthetic neighbours.
    """

    if neighbors < 1:
        raise ValueError("neighbors must be positive.")
    dimension = reference.shape[1]
    reference_n = reference.shape[0]
    bandwidth = (reference_n * (dimension + 2.0) / 4.0) ** (
        -1.0 / (dimension + 4.0)
    )
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(reference)
    neighbor_model = NearestNeighbors(
        n_neighbors=min(neighbors, len(synthetic)),
        metric="euclidean",
    ).fit(synthetic)
    _, indices = neighbor_model.kneighbors(query)
    local_synthetic = synthetic[indices]
    flattened = local_synthetic.reshape(-1, dimension)
    baseline_log_density = kde.score_samples(flattened).reshape(indices.shape)
    squared_distance = np.square(local_synthetic - query[:, None, :]).sum(axis=2)
    log_kernel = (
        -0.5 * dimension * log(2.0 * pi)
        - dimension * log(bandwidth)
        - squared_distance / (2.0 * bandwidth * bandwidth)
    )
    augmented_log_density = np.logaddexp(
        log(reference_n) + baseline_log_density,
        log_kernel,
    ) - log(reference_n + 1.0)
    return (augmented_log_density - baseline_log_density).sum(axis=1)


def tpr_at_fpr(
    labels: Sequence[int],
    scores: Sequence[float],
    target_fpr: float,
) -> tuple[float, float]:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    nonmember_scores = scores_array[labels_array == 0]
    member_scores = scores_array[labels_array == 1]
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must lie strictly between zero and one.")
    if len(nonmember_scores) == 0 or len(member_scores) == 0:
        raise ValueError("Both member and nonmember records are required.")
    threshold = np.quantile(
        nonmember_scores,
        1.0 - target_fpr,
        method="higher",
    )
    realized_fpr = float((nonmember_scores >= threshold).mean())
    tpr = float((member_scores >= threshold).mean())
    return tpr, realized_fpr


def attack_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if not np.isfinite(scores_array).all():
        raise ValueError("Attack scores must be finite.")
    tpr_1, fpr_1 = tpr_at_fpr(labels_array, scores_array, 0.01)
    tpr_5, fpr_5 = tpr_at_fpr(labels_array, scores_array, 0.05)
    return {
        "AUC": float(roc_auc_score(labels_array, scores_array)),
        "TPR1FPR": tpr_1,
        "RealizedFPR1": fpr_1,
        "TPR5FPR": tpr_5,
        "RealizedFPR5": fpr_5,
    }


def bootstrap_attack_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    repetitions: int,
    seed: int,
    confidence_level: float = 0.95,
) -> Mapping[str, tuple[float, float]]:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if repetitions < 1:
        raise ValueError("repetitions must be positive.")
    member_indices = np.flatnonzero(labels_array == 1)
    nonmember_indices = np.flatnonzero(labels_array == 0)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(repetitions):
        sampled_members = rng.choice(member_indices, len(member_indices), replace=True)
        sampled_nonmembers = rng.choice(
            nonmember_indices,
            len(nonmember_indices),
            replace=True,
        )
        indices = np.concatenate([sampled_members, sampled_nonmembers])
        rows.append(attack_metrics(labels_array[indices], scores_array[indices]))
    frame = pd.DataFrame(rows)
    tail = (1.0 - confidence_level) / 2.0
    return {
        metric: (
            float(frame[metric].quantile(tail)),
            float(frame[metric].quantile(1.0 - tail)),
        )
        for metric in ("AUC", "TPR1FPR", "TPR5FPR")
    }
