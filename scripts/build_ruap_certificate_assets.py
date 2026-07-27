from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import build_ruap_audit as ruap


def refresh_statuses(report: pd.DataFrame) -> pd.DataFrame:
    refreshed = report.copy()
    refreshed["Status"] = refreshed.apply(
        ruap.classify_status,
        axis=1,
        auc_tol=0.010,
        ece_tol=0.005,
        cost_tol=0.010,
        exposure_tol=0.010,
    )
    return refreshed


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh RUA-P audit-profile assets from an existing report CSV.")
    parser.add_argument("--report-csv", default="paper_assets/ruap_audit/ruap_report_card.csv")
    parser.add_argument("--output-dir", default="paper_assets/ruap_audit")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = pd.read_csv(args.report_csv)
    report = refresh_statuses(report)
    report.to_csv(output_dir / "ruap_report_card.csv", index=False)
    ruap.write_latex_tables(report, output_dir)

    profile = ruap.build_ruap_certificate(report)
    profile.to_csv(output_dir / "ruap_certificate.csv", index=False)
    ruap.write_certificate_table(profile, output_dir)

    sensitivity = ruap.build_threshold_sensitivity(report)
    sensitivity.to_csv(output_dir / "ruap_threshold_sensitivity.csv", index=False)
    ruap.write_threshold_sensitivity_table(sensitivity, output_dir)

    print(f"refreshed {len(report)} report rows and {len(profile)} decision-profile rows in {output_dir}")


if __name__ == "__main__":
    main()
