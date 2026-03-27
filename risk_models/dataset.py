from typing import Any, Dict

from risk_models.configs import DatasetConfig
from risk_models.datasets.australian_credit import load_australian_credit
from risk_models.datasets.give_me_some_credit import load_give_me_some_credit
from risk_models.datasets.german_credit import load_german_credit
from risk_models.datasets.taiwan_default import load_taiwan_default


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
        return load_give_me_some_credit(dataset_config)
    if dataset_config.name == "taiwan_default":
        return load_taiwan_default(dataset_config)
    if dataset_config.name == "australian_credit":
        return load_australian_credit(dataset_config)
    raise ValueError(f"Unknown dataset: {dataset_config.name}")


def load_dataset_from_config(dataset_config: DatasetConfig) -> Dict[str, Any]:
    return load_dataset(dataset_config)


def preprocess_data(filepath: str):
    bundle = load_german_credit(path=filepath, top_frac=0.30)
    metadata = bundle["metadata"]
    return bundle["X"], bundle["y"], metadata["numeric_cols"]
