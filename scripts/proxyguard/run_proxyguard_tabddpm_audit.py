from __future__ import annotations

import argparse
import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.proxyguard.tabular.config import get_dataset_config  # noqa: E402
from scripts.proxyguard.tabular.data import load_dataset  # noqa: E402
from proxyguard.core import (  # noqa: E402
    RiskRequirement,
    audit_proxy_candidates,
    paired_prediction_losses,
)
from scripts.proxyguard.run_proxyguard_aim_audit import (  # noqa: E402
    _mean_normalized_cost,
    fit_selected_procedure,
    make_split,
)

REQUIREMENTS = [
    RiskRequirement("Brier", tolerance=0.01, lower=-1.0, upper=1.0),
    RiskRequirement("Clipped log loss", tolerance=0.01, lower=-1.0, upper=1.0),
    RiskRequirement("Cost5x", tolerance=0.01, lower=-1.0, upper=1.0),
]


def _import_official_tabddpm(repository: Path):
    repository = repository.resolve()
    scripts_path = repository / "scripts"
    for path in (str(scripts_path), str(repository)):
        if path not in sys.path:
            sys.path.insert(0, path)

    def _no_op_install() -> None:
        return None

    icecream_stub = types.ModuleType("icecream")
    icecream_stub.install = _no_op_install
    sys.modules["icecream"] = icecream_stub
    from sample import sample as official_sample
    from train import train as official_train

    return official_train, official_sample


def _save_official_dataset(
    path: Path,
    split,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    arrays = {
        "train": (split.X_train, split.y_train),
        "val": (split.X_validation, split.y_validation),
        "test": (split.X_audit, split.y_audit),
    }
    for name, (X_part, y_part) in arrays.items():
        np.save(path / f"X_num_{name}.npy", X_part.to_numpy(dtype=np.float32))
        np.save(path / f"y_{name}.npy", y_part.to_numpy(dtype=np.int64))
    (path / "info.json").write_text(
        json.dumps(
            {
                "name": path.name,
                "id": path.name,
                "task_type": "binclass",
                "n_classes": 2,
                "n_num_features": split.X_train.shape[1],
                "n_cat_features": 0,
                "train_size": len(split.y_train),
                "val_size": len(split.y_validation),
                "test_size": len(split.y_audit),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _save_attack_split(
    X: pd.DataFrame,
    y: pd.Series,
    path: Path,
) -> None:
    frame = X.reset_index(drop=True).copy()
    frame["__target__"] = y.reset_index(drop=True).to_numpy(dtype=int)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_tabddpm_audit(
    registry: dict,
    repository: Path,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import torch

    setting = registry["modern_generators"]["tabddpm"]
    official_train, official_sample = _import_official_tabddpm(repository)
    repository_commit = (
        setting["commit"]
        if str(setting["commit"])
        else "unregistered"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    candidate_regrets: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: list[dict[str, float | int | str]] = []
    loss_frames: list[pd.DataFrame] = []

    for dataset_offset, dataset_name in enumerate(setting["datasets"]):
        bundle = load_dataset(get_dataset_config(dataset_name))
        X = bundle["X"].reset_index(drop=True).astype(float)
        y = pd.Series(bundle["y"]).reset_index(drop=True).astype(int)
        split_seed = int(setting["seed"]) + dataset_offset * 101
        split = make_split(X, y, split_seed)
        (
            source_model_name,
            source_model,
            source_threshold,
            _source_validation_probability,
        ) = fit_selected_procedure(
            split.X_train,
            split.y_train,
            split.X_validation,
            split.y_validation,
            seed=split_seed,
        )
        source_audit_probability = source_model.predict_proba(split.X_audit)[:, 1]

        dataset_root = output_root / "official_data" / dataset_name
        model_root = output_root / "models" / dataset_name
        model_root.mkdir(parents=True, exist_ok=True)
        _save_official_dataset(dataset_root, split)
        split_root = output_root / "splits" / dataset_name
        _save_attack_split(split.X_train, split.y_train, split_root / "members.csv")
        _save_attack_split(
            split.X_validation,
            split.y_validation,
            split_root / "reference.csv",
        )
        _save_attack_split(split.X_audit, split.y_audit, split_root / "nonmembers.csv")

        model_params = {
            "num_classes": 2,
            "is_y_cond": True,
            "rtdl_params": {
                "d_layers": [int(value) for value in setting["model"]["d_layers"]],
                "dropout": float(setting["model"]["dropout"]),
            },
            "dim_t": int(setting["model"]["dim_t"]),
        }
        transformations = {
            "seed": int(setting["seed"]),
            "normalization": str(setting["normalization"]),
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default",
        }
        official_train(
            parent_dir=str(model_root),
            real_data_path=str(dataset_root),
            steps=int(setting["training_steps"]),
            lr=float(setting["learning_rate"]),
            weight_decay=float(setting["weight_decay"]),
            batch_size=min(int(setting["batch_size"]), len(split.y_train)),
            model_type=str(setting["model"]["type"]),
            model_params=copy.deepcopy(model_params),
            num_timesteps=int(setting["diffusion_steps"]),
            gaussian_loss_type="mse",
            scheduler="cosine",
            T_dict=transformations,
            num_numerical_features=X.shape[1],
            device=device,
            seed=int(setting["seed"]) + dataset_offset,
            change_val=False,
        )
        official_sample(
            parent_dir=str(model_root),
            real_data_path=str(dataset_root),
            batch_size=min(10_000, len(split.y_train)),
            num_samples=len(split.y_train),
            model_type=str(setting["model"]["type"]),
            model_params=copy.deepcopy(model_params),
            model_path=str(model_root / "model_ema.pt"),
            num_timesteps=int(setting["diffusion_steps"]),
            gaussian_loss_type="mse",
            scheduler="cosine",
            T_dict=transformations,
            num_numerical_features=X.shape[1],
            disbalance=None,
            device=device,
            seed=int(setting["seed"]) + dataset_offset,
            change_val=False,
        )

        synthetic_X = pd.DataFrame(
            np.load(model_root / "X_num_train.npy"),
            columns=X.columns,
        ).astype(float)
        synthetic_y = pd.Series(
            np.load(model_root / "y_train.npy"),
            name="__target__",
        ).round().clip(0, 1).astype(int)
        if synthetic_y.nunique() < 2:
            raise RuntimeError(f"{dataset_name} TabDDPM release has one target class.")
        release_frame = synthetic_X.copy()
        release_frame["__target__"] = synthetic_y.to_numpy()
        release_path = output_root / "releases" / f"{dataset_name}.csv"
        release_path.parent.mkdir(parents=True, exist_ok=True)
        release_frame.to_csv(release_path, index=False)

        (
            proxy_model_name,
            proxy_model,
            proxy_threshold,
            _proxy_validation_probability,
        ) = fit_selected_procedure(
            synthetic_X,
            synthetic_y,
            split.X_validation,
            split.y_validation,
            seed=int(setting["seed"]) + dataset_offset,
        )
        proxy_audit_probability = proxy_model.predict_proba(split.X_audit)[:, 1]
        candidate = f"{dataset_name}::tabddpm"
        losses = paired_prediction_losses(
            y_true=split.y_audit,
            source_probability=source_audit_probability,
            proxy_probability=proxy_audit_probability,
            source_thresholds={5.0: source_threshold},
            proxy_thresholds={5.0: proxy_threshold},
            record_ids=split.audit_positions,
        )
        losses.insert(0, "Candidate", candidate)
        loss_frames.append(losses)
        candidate_regrets[candidate] = {
            "Brier": losses["brier_regret"].to_numpy(),
            "Clipped log loss": losses["logloss_regret"].to_numpy(),
            "Cost5x": losses["cost5x_regret"].to_numpy(),
        }
        diagnostics.append(
            {
                "Candidate": candidate,
                "Dataset": dataset_name,
                "Mechanism": "TabDDPM",
                "RepositoryCommit": repository_commit,
                "AuditN": len(split.y_audit),
                "SourceModel": source_model_name,
                "ProxyModel": proxy_model_name,
                "SourceAUC": roc_auc_score(split.y_audit, source_audit_probability),
                "ProxyAUC": roc_auc_score(split.y_audit, proxy_audit_probability),
                "AUCChange": roc_auc_score(split.y_audit, proxy_audit_probability)
                - roc_auc_score(split.y_audit, source_audit_probability),
                "SourceCost5x": _mean_normalized_cost(
                    split.y_audit,
                    source_audit_probability,
                    source_threshold,
                    5.0,
                ),
                "ProxyCost5x": _mean_normalized_cost(
                    split.y_audit,
                    proxy_audit_probability,
                    proxy_threshold,
                    5.0,
                ),
                "SyntheticPositiveRate": float(synthetic_y.mean()),
                "TrainingSteps": int(setting["training_steps"]),
                "DiffusionSteps": int(setting["diffusion_steps"]),
                "Device": str(device),
            }
        )

    result = audit_proxy_candidates(
        candidate_regrets,
        requirements=REQUIREMENTS,
        alpha=float(registry["risk_control"]["alpha"]),
        bound_method=str(registry["risk_control"]["bound_method"]),
    )
    summary = result.candidate_summary.merge(
        pd.DataFrame(diagnostics),
        on="Candidate",
        how="left",
        validate="one_to_one",
    )
    return summary, result.requirement_detail, pd.concat(loss_frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the registered official-implementation TabDDPM proxy audit."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_frontier_registry.json",
    )
    parser.add_argument(
        "--tabddpm-repo",
        default=r"C:\tmp\tabddpm-official",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_tabddpm",
    )
    args = parser.parse_args()
    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    repository = Path(args.tabddpm_repo)
    if not (repository / "tab_ddpm").exists():
        raise FileNotFoundError(f"Official TabDDPM repository not found: {repository}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary, detail, losses = run_tabddpm_audit(
        registry,
        repository,
        output_root,
    )
    summary.to_csv(output_root / "candidate_summary.csv", index=False)
    detail.to_csv(output_root / "requirement_detail.csv", index=False)
    losses.to_csv(output_root / "paired_losses.csv", index=False)
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "registered_tabddpm": registry["modern_generators"]["tabddpm"],
                "repository": str(repository),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        summary[
            ["Candidate", "AuditN", "Status", "AUCChange", "SourceCost5x", "ProxyCost5x"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
