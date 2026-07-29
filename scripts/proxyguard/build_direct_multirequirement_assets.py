from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAMED = "Named-release Holm"
DIRECT = "Direct shared-target"


def _power_gain(path: Path, reliability: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[
        (frame["TrueReliability"] == reliability)
        & frame["Method"].isin([NAMED, DIRECT])
    ]
    pivot = selected.pivot_table(
        index="Releases",
        columns=["TargetN", "Method"],
        values="ValidationRate",
    )
    releases = sorted(selected["Releases"].unique())
    target_sizes = sorted(selected["TargetN"].unique())
    values = np.empty((len(releases), len(target_sizes)), dtype=float)
    for row, release_count in enumerate(releases):
        for column, target_size in enumerate(target_sizes):
            values[row, column] = 100.0 * (
                pivot.loc[release_count, (target_size, DIRECT)]
                - pivot.loc[release_count, (target_size, NAMED)]
            )
    return pd.DataFrame(values, index=releases, columns=target_sizes)


def build_figure(
    *,
    moderate_summary: Path,
    high_signal_summary: Path,
    output: Path,
) -> None:
    panels = (
        ("Moderate evidence", _power_gain(moderate_summary, 0.95)),
        ("High-signal releases", _power_gain(high_signal_summary, 0.95)),
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.35), constrained_layout=True)
    image = None
    for axis, (title, frame) in zip(axes, panels, strict=True):
        image = axis.imshow(
            frame.to_numpy(),
            cmap="RdBu",
            vmin=-100.0,
            vmax=100.0,
            aspect="auto",
        )
        axis.set_title(title, fontsize=10, pad=7)
        axis.set_xlabel("Shared target records $n$")
        axis.set_xticks(range(frame.shape[1]), [f"{value:,}" for value in frame.columns])
        axis.set_yticks(range(frame.shape[0]), [f"{value:,}" for value in frame.index])
        axis.set_ylabel("Mechanism draws $R$")
        for row in range(frame.shape[0]):
            for column in range(frame.shape[1]):
                value = float(frame.iloc[row, column])
                text_color = "white" if abs(value) >= 55.0 else "#202020"
                axis.text(
                    column,
                    row,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
        for spine in axis.spines.values():
            spine.set_visible(False)
    assert image is not None
    colorbar = figure.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        shrink=0.78,
        pad=0.08,
        aspect=35,
    )
    colorbar.set_label("Direct minus named power (percentage points)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the fair shared-target comparison figure."
    )
    parser.add_argument(
        "--moderate-summary",
        type=Path,
        default=Path(
            "outputs/proxyguard_direct_multirequirement_moderate_confirmatory/summary.csv"
        ),
    )
    parser.add_argument(
        "--high-signal-summary",
        type=Path,
        default=Path(
            "outputs/proxyguard_direct_multirequirement_confirmatory/summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/proxyguard/figs/direct_multirequirement_power.png"),
    )
    args = parser.parse_args()
    build_figure(
        moderate_summary=args.moderate_summary,
        high_signal_summary=args.high_signal_summary,
        output=args.output,
    )


if __name__ == "__main__":
    main()
