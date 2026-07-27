from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proxyguard.attacks import (  # noqa: E402
    attack_metrics,
    bootstrap_attack_metrics,
    dcr_scores,
    domias_kde_scores,
    fit_attack_representation,
    gen_lra_scores,
)
from proxyguard.core import RiskRequirement, audit_proxy_candidates  # noqa: E402


@dataclass(frozen=True)
class Release:
    generator: str
    dataset: str
    setting: str
    path: Path
    split_root: Path

    @property
    def name(self) -> str:
        return f"{self.dataset}::{self.generator}::{self.setting}"


def discover_releases(
    aim_root: Path,
    tabddpm_root: Path,
) -> list[Release]:
    releases: list[Release] = []
    aim_release_pattern = re.compile(r"aim_e(.+)_release_1$")
    aim_release_root = aim_root / "releases"
    if aim_release_root.exists():
        for path in sorted(aim_release_root.glob("*/*_release_1.csv")):
            match = aim_release_pattern.match(path.stem)
            if match is None:
                continue
            dataset = path.parent.name
            releases.append(
                Release(
                    generator="AIM",
                    dataset=dataset,
                    setting=f"epsilon={match.group(1)}",
                    path=path,
                    split_root=aim_root / "splits" / dataset,
                )
            )
    tabddpm_release_root = tabddpm_root / "releases"
    if tabddpm_release_root.exists():
        for path in sorted(tabddpm_release_root.glob("*.csv")):
            dataset = path.stem
            releases.append(
                Release(
                    generator="TabDDPM",
                    dataset=dataset,
                    setting="registered",
                    path=path,
                    split_root=tabddpm_root / "splits" / dataset,
                )
            )
    if not releases:
        raise FileNotFoundError("No registered AIM or TabDDPM releases were found.")
    return releases


def _sample_frame(
    frame: pd.DataFrame,
    limit: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.reset_index(drop=True)
    positions = rng.choice(len(frame), size=limit, replace=False)
    return frame.iloc[np.sort(positions)].reset_index(drop=True)


def _read_aligned_release(release: Release) -> tuple[pd.DataFrame, ...]:
    members = pd.read_csv(release.split_root / "members.csv")
    nonmembers = pd.read_csv(release.split_root / "nonmembers.csv")
    reference = pd.read_csv(release.split_root / "reference.csv")
    synthetic = pd.read_csv(release.path)
    columns = list(reference.columns)
    for name, frame in (
        ("members", members),
        ("nonmembers", nonmembers),
        ("synthetic", synthetic),
    ):
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{release.name} {name} table is missing columns: {sorted(missing)}"
            )
    return (
        members[columns],
        nonmembers[columns],
        reference[columns],
        synthetic[columns],
    )


def run_attack_suite(
    releases: list[Release],
    setting: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    limits = setting["subsample_limits"]
    attacker_seeds = [int(value) for value in setting["attacker_seeds"]]
    pca_dimensions = int(setting["representation"]["pca_dimensions"])
    jitter = float(setting["representation"]["kde_jitter"])
    neighbors = next(
        int(attack["neighbors"])
        for attack in setting["attacks"]
        if attack["name"] == "Gen-LRA"
    )
    detail_rows: list[dict[str, float | int | str]] = []
    score_frames: list[pd.DataFrame] = []

    for release in releases:
        full_members, full_nonmembers, full_reference, full_synthetic = (
            _read_aligned_release(release)
        )
        for attacker_seed in attacker_seeds:
            rng = np.random.default_rng(attacker_seed)
            members = _sample_frame(
                full_members,
                int(limits["members"]),
                rng,
            )
            nonmembers = _sample_frame(
                full_nonmembers,
                int(limits["nonmembers"]),
                rng,
            )
            reference = _sample_frame(
                full_reference,
                int(limits["reference"]),
                rng,
            )
            synthetic = _sample_frame(
                full_synthetic,
                int(limits["synthetic"]),
                rng,
            )
            representation = fit_attack_representation(
                reference,
                pca_dimensions=pca_dimensions,
            )
            member_array = representation.transform(members)
            nonmember_array = representation.transform(nonmembers)
            reference_array = representation.transform(reference)
            synthetic_array = representation.transform(synthetic)
            query = np.vstack([member_array, nonmember_array])
            labels = np.concatenate(
                [
                    np.ones(len(member_array), dtype=int),
                    np.zeros(len(nonmember_array), dtype=int),
                ]
            )
            domias, density_only = domias_kde_scores(
                query,
                synthetic_array,
                reference_array,
                seed=attacker_seed,
                jitter=jitter,
            )
            score_map = {
                "DOMIAS-KDE": domias,
                "Gen-LRA": gen_lra_scores(
                    query,
                    synthetic_array,
                    reference_array,
                    neighbors=neighbors,
                ),
                "density-only KDE": density_only,
                "DCR": dcr_scores(query, synthetic_array),
            }
            for attack_name, scores in score_map.items():
                metrics = attack_metrics(labels, scores)
                intervals = bootstrap_attack_metrics(
                    labels,
                    scores,
                    repetitions=int(setting["bootstrap_repetitions"]),
                    seed=int(setting["bootstrap_seed"]) + attacker_seed,
                    confidence_level=float(setting["confidence_level"]),
                )
                detail_rows.append(
                    {
                        "Release": release.name,
                        "Generator": release.generator,
                        "Dataset": release.dataset,
                        "Setting": release.setting,
                        "AttackerSeed": attacker_seed,
                        "Attack": attack_name,
                        "Members": len(member_array),
                        "Nonmembers": len(nonmember_array),
                        "ReferenceN": len(reference_array),
                        "SyntheticN": len(synthetic_array),
                        **metrics,
                        "AUCBootstrapLow": intervals["AUC"][0],
                        "AUCBootstrapHigh": intervals["AUC"][1],
                        "TPR1BootstrapLow": intervals["TPR1FPR"][0],
                        "TPR1BootstrapHigh": intervals["TPR1FPR"][1],
                        "TPR5BootstrapLow": intervals["TPR5FPR"][0],
                        "TPR5BootstrapHigh": intervals["TPR5FPR"][1],
                    }
                )
                score_frames.append(
                    pd.DataFrame(
                        {
                            "Release": release.name,
                            "AttackerSeed": attacker_seed,
                            "Attack": attack_name,
                            "Membership": labels,
                            "Score": scores,
                        }
                    )
                )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(
            ["Release", "Generator", "Dataset", "Setting", "Attack"],
            as_index=False,
        )
        .agg(
            AttackerSeeds=("AttackerSeed", "nunique"),
            AUCMean=("AUC", "mean"),
            AUCSeedLow=("AUC", lambda values: float(np.quantile(values, 0.025))),
            AUCSeedHigh=("AUC", lambda values: float(np.quantile(values, 0.975))),
            AUCMeanBootstrapLow=("AUCBootstrapLow", "mean"),
            AUCMeanBootstrapHigh=("AUCBootstrapHigh", "mean"),
            TPR1Mean=("TPR1FPR", "mean"),
            TPR1SeedLow=("TPR1FPR", lambda values: float(np.quantile(values, 0.025))),
            TPR1SeedHigh=("TPR1FPR", lambda values: float(np.quantile(values, 0.975))),
            TPR5Mean=("TPR5FPR", "mean"),
            TPR5SeedLow=("TPR5FPR", lambda values: float(np.quantile(values, 0.025))),
            TPR5SeedHigh=("TPR5FPR", lambda values: float(np.quantile(values, 0.975))),
        )
        .sort_values(["Generator", "Dataset", "Setting", "Attack"], ignore_index=True)
    )
    return summary, pd.concat(score_frames, ignore_index=True)


def build_attack_advantage_audit(
    scores: pd.DataFrame,
    setting: dict,
    alpha: float,
    bound_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    claim = setting["risk_controlled_claim"]
    fixed_seed = int(claim["attacker_seed"])
    calibration_fraction = float(claim["nonmember_threshold_fraction"])
    target_fpr = float(claim["threshold_calibration_fpr"])
    ceiling = float(claim["attack_advantage_ceiling"])
    fixed = scores[scores["AttackerSeed"] == fixed_seed]
    candidate_regrets: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: list[dict[str, float | int | str]] = []

    for (release, attack), group in fixed.groupby(["Release", "Attack"], sort=True):
        member_scores = group.loc[group["Membership"] == 1, "Score"].to_numpy()
        nonmember_scores = group.loc[group["Membership"] == 0, "Score"].to_numpy()
        calibration_n = round(calibration_fraction * len(nonmember_scores))
        calibration_n = min(max(calibration_n, 1), len(nonmember_scores) - 1)
        threshold_scores = nonmember_scores[:calibration_n]
        audit_nonmember_scores = nonmember_scores[calibration_n:]
        threshold = np.quantile(
            threshold_scores,
            1.0 - target_fpr,
            method="higher",
        )
        pair_n = min(len(member_scores), len(audit_nonmember_scores))
        member_detection = (member_scores[:pair_n] >= threshold).astype(float)
        nonmember_detection = (
            audit_nonmember_scores[:pair_n] >= threshold
        ).astype(float)
        advantage = member_detection - nonmember_detection
        candidate_regrets.setdefault(release, {})[attack] = advantage
        diagnostics.append(
            {
                "Release": release,
                "Attack": attack,
                "N": pair_n,
                "Threshold": float(threshold),
                "TPR": float(member_detection.mean()),
                "FPR": float(nonmember_detection.mean()),
                "AttackAdvantage": float(advantage.mean()),
            }
        )

    attack_names = sorted(fixed["Attack"].unique())
    requirements = [
        RiskRequirement(
            name=attack,
            tolerance=ceiling,
            lower=-1.0,
            upper=1.0,
        )
        for attack in attack_names
    ]
    complete = {
        release: attack_map
        for release, attack_map in candidate_regrets.items()
        if set(attack_map) == set(attack_names)
    }
    result = audit_proxy_candidates(
        complete,
        requirements=requirements,
        alpha=alpha,
        bound_method=bound_method,
    )
    return result.candidate_summary, result.requirement_detail, pd.DataFrame(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the registered no-box membership attack suite."
    )
    parser.add_argument(
        "--registry",
        default="registries/proxyguard_frontier_registry.json",
    )
    parser.add_argument(
        "--aim-root",
        default="outputs/proxyguard_repeated_aim",
    )
    parser.add_argument(
        "--tabddpm-root",
        default="outputs/proxyguard_tabddpm",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/proxyguard_privacy_attacks",
    )
    args = parser.parse_args()
    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    releases = discover_releases(Path(args.aim_root), Path(args.tabddpm_root))
    summary, scores = run_attack_suite(releases, registry["attack_suite"])
    advantage_summary, advantage_detail, advantage_diagnostics = (
        build_attack_advantage_audit(
            scores,
            registry["attack_suite"],
            alpha=float(registry["risk_control"]["alpha"]),
            bound_method=str(
                registry["attack_suite"]["risk_controlled_claim"]["bound_method"]
            ),
        )
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "attack_summary.csv", index=False)
    scores.to_csv(output_root / "attack_scores.csv", index=False)
    advantage_summary.to_csv(
        output_root / "attack_advantage_summary.csv",
        index=False,
    )
    advantage_detail.to_csv(
        output_root / "attack_advantage_detail.csv",
        index=False,
    )
    advantage_diagnostics.to_csv(
        output_root / "attack_advantage_diagnostics.csv",
        index=False,
    )
    (output_root / "settings.json").write_text(
        json.dumps(
            {
                "registry": str(registry_path),
                "release_count": len(releases),
                "registered_attack_suite": registry["attack_suite"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        summary[
            ["Generator", "Dataset", "Setting", "Attack", "AUCMean", "TPR1Mean", "TPR5Mean"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
