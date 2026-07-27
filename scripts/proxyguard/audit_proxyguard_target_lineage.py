from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from risk_models.configs import get_dataset_config  # noqa: E402
from risk_models.dataset import load_dataset  # noqa: E402


@dataclass(frozen=True)
class SplitPositions:
    train: np.ndarray
    validation: np.ndarray
    audit: np.ndarray


def make_split_positions(y: pd.Series, seed: int) -> SplitPositions:
    positions = np.arange(len(y))
    train, holdout = train_test_split(
        positions,
        test_size=0.40,
        random_state=seed,
        stratify=y,
    )
    validation, audit = train_test_split(
        holdout,
        test_size=0.50,
        random_state=seed + 1,
        stratify=y.iloc[holdout],
    )
    return SplitPositions(
        train=np.asarray(train, dtype=int),
        validation=np.asarray(validation, dtype=int),
        audit=np.asarray(audit, dtype=int),
    )


def summarize_overlap(
    y: pd.Series,
    *,
    study: str,
    pilot_seed: int,
    replication_seed: int,
    pilot_informed_choice: str,
) -> tuple[dict[str, int | str | bool], pd.DataFrame]:
    pilot = make_split_positions(y, pilot_seed)
    replication = make_split_positions(y, replication_seed)
    audit = set(replication.audit.tolist())
    pilot_roles = {
        "train": set(pilot.train.tolist()),
        "validation": set(pilot.validation.tolist()),
        "audit": set(pilot.audit.tolist()),
    }
    overlaps = {
        role: len(audit & positions)
        for role, positions in pilot_roles.items()
    }
    any_overlap = len(audit & set.union(*pilot_roles.values()))
    summary: dict[str, int | str | bool] = {
        "Study": study,
        "PilotSplitSeed": pilot_seed,
        "ReplicationSplitSeed": replication_seed,
        "ReplicationAuditN": len(replication.audit),
        "OverlapPilotTrain": overlaps["train"],
        "OverlapPilotValidation": overlaps["validation"],
        "OverlapPilotAudit": overlaps["audit"],
        "OverlapAnyPilotRole": any_overlap,
        "PilotInformedChoice": pilot_informed_choice,
        "RecordDisjointFromPilot": any_overlap == 0,
        "TheoremBackedProspectiveClaim": any_overlap == 0,
    }

    role_by_position = {
        position: role
        for role, positions in pilot_roles.items()
        for position in positions
    }
    detail = pd.DataFrame(
        {
            "Study": study,
            "ReplicationAuditPosition": replication.audit,
            "PilotRole": [
                role_by_position[int(position)]
                for position in replication.audit
            ],
        }
    ).sort_values("ReplicationAuditPosition", ignore_index=True)
    return summary, detail


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit record overlap between ProxyGuard pilots and informed replications."
    )
    parser.add_argument(
        "--pilot-registry",
        default="registries/proxyguard_bootstrap_mechanism_registry.json",
    )
    parser.add_argument(
        "--aim-registry",
        default="registries/proxyguard_mechanism_revision_registry.json",
    )
    parser.add_argument(
        "--bootstrap-registry",
        default="registries/proxyguard_bootstrap_positive_replication_registry.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_target_lineage",
    )
    args = parser.parse_args()

    paths = {
        "pilot_registry": Path(args.pilot_registry),
        "aim_registry": Path(args.aim_registry),
        "bootstrap_registry": Path(args.bootstrap_registry),
    }
    registries = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    dataset_name = str(registries["pilot_registry"]["dataset"])
    if dataset_name != str(registries["bootstrap_registry"]["dataset"]):
        raise ValueError("Pilot and bootstrap replication must use the same dataset.")
    aim_dataset = str(
        registries["aim_registry"]["prospective_aim_mechanism"]["dataset"]
    )
    if dataset_name != aim_dataset:
        raise ValueError("Pilot and AIM replication must use the same dataset.")

    bundle = load_dataset(get_dataset_config(dataset_name))
    y = pd.Series(bundle["y"]).reset_index(drop=True).astype(int)
    pilot_seed = int(registries["pilot_registry"]["split_seed"])
    aim_datasets = list(registries["aim_registry"]["modern_generators"]["aim"]["datasets"])
    aim_offset = aim_datasets.index(aim_dataset)
    aim_seed = (
        int(registries["aim_registry"]["modern_generators"]["aim"]["release_seeds"][0])
        + 101 * aim_offset
    )
    bootstrap_seed = int(registries["bootstrap_registry"]["split_seed"])

    rows: list[dict[str, int | str | bool]] = []
    details: list[pd.DataFrame] = []
    for study, replication_seed, choice in [
        (
            "AIM informed replication",
            aim_seed,
            "AIM epsilon=1 configuration",
        ),
        (
            "Bootstrap informed positive control",
            bootstrap_seed,
            "relative and absolute limits",
        ),
    ]:
        summary, detail = summarize_overlap(
            y,
            study=study,
            pilot_seed=pilot_seed,
            replication_seed=replication_seed,
            pilot_informed_choice=choice,
        )
        rows.append(summary)
        details.append(detail)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_root / "lineage_summary.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(
        output_root / "lineage_records.csv",
        index=False,
    )
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "records": len(y),
                "registries": {
                    name: {
                        "path": str(path),
                        "sha256": _sha256(path),
                    }
                    for name, path in paths.items()
                },
                "interpretation": (
                    "The informed replication audits reuse records that appeared "
                    "in pilot train, validation, or audit roles. Their numerical "
                    "bounds are sensitivity summaries, not theorem-backed "
                    "prospective guarantees."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
