from typing import Any, Dict

import numpy as np
import pandas as pd

from scripts.proxyguard.tabular.config import DatasetConfig
from scripts.proxyguard.tabular.datasets.australian_credit import load_australian_credit
from scripts.proxyguard.tabular.datasets.generic_tabular import load_generic_tabular
from scripts.proxyguard.tabular.datasets.german_credit import load_german_credit
from scripts.proxyguard.tabular.datasets.give_me_some_credit import load_give_me_some_credit
from scripts.proxyguard.tabular.datasets.taiwan_default import load_taiwan_default


def load_dataset(dataset_config: DatasetConfig) -> Dict[str, Any]:
    if dataset_config.name == "german_credit":
        return load_german_credit(
            path=dataset_config.path,
            top_frac=dataset_config.label_params.get("top_frac", 0.30),
            threshold_shift=dataset_config.label_params.get("threshold_shift", 0.0),
            noise_rate=dataset_config.label_params.get("noise_rate", 0.0),
            random_state=dataset_config.label_params.get("random_state", 3407),
            drop_columns=dataset_config.drop_columns,
        )
    if dataset_config.name == "give_me_some_credit":
        return _apply_generic_label_noise(load_give_me_some_credit(dataset_config), dataset_config)
    if dataset_config.name == "taiwan_default":
        return _apply_generic_label_noise(load_taiwan_default(dataset_config), dataset_config)
    if dataset_config.name == "australian_credit":
        return _apply_generic_label_noise(load_australian_credit(dataset_config), dataset_config)
    if dataset_config.label_params.get("loader") == "generic_tabular":
        return _apply_generic_label_noise(load_generic_tabular(dataset_config), dataset_config)
    raise ValueError(f"Unknown dataset: {dataset_config.name}")


def _apply_generic_label_noise(bundle: Dict[str, Any], dataset_config: DatasetConfig) -> Dict[str, Any]:
    noise_rate = float(dataset_config.label_params.get("noise_rate", 0.0))
    if noise_rate <= 0.0:
        return bundle

    noisy_bundle = dict(bundle)
    y = pd.Series(bundle["y"]).copy()
    rng = np.random.default_rng(int(dataset_config.label_params.get("random_state", 3407)))
    n_flip = int(round(len(y) * noise_rate))
    if n_flip > 0:
        flip_positions = rng.choice(len(y), size=n_flip, replace=False)
        y_array = y.to_numpy(copy=True).astype(int)
        y_array[flip_positions] = 1 - y_array[flip_positions]
        noisy_bundle["y"] = pd.Series(y_array, index=y.index, name=y.name)

    metadata = dict(bundle.get("metadata", {}))
    label_params = dict(metadata.get("label_params", {}))
    label_params.update(dataset_config.label_params)
    metadata["label_params"] = label_params
    metadata["label_type"] = f"{metadata.get('label_type', dataset_config.label_type)}+synthetic_noise"
    noisy_bundle["metadata"] = metadata
    return noisy_bundle


def load_dataset_from_config(dataset_config: DatasetConfig) -> Dict[str, Any]:
    return load_dataset(dataset_config)


def preprocess_data(filepath: str):
    bundle = load_german_credit(path=filepath, top_frac=0.30)
    metadata = bundle["metadata"]
    return bundle["X"], bundle["y"], metadata["numeric_cols"]
