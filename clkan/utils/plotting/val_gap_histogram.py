from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_gap_dataframe(results_path: Path, metric: str) -> pd.DataFrame:
    with results_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    rows = []
    for entry in results:
        extra_infos = entry.get("extra_infos", {})
        val_losses = entry.get("val_losses", {})
        for fold, info in extra_infos.items():
            if not isinstance(info, dict):
                continue
            best_seen_val = info.get("best_seen_val_loss")
            final_val = val_losses.get(str(fold), {}).get(metric)
            if final_val is None:
                final_val = val_losses.get(fold, {}).get(metric)
            if best_seen_val is None or final_val is None:
                continue

            if abs(float(best_seen_val)) < 1e-12:
                relative_gap_pct = np.nan
            else:
                relative_gap_pct = 100.0 * (
                    float(final_val) - float(best_seen_val)
                ) / abs(float(best_seen_val))

            rows.append(
                {
                    "dataset_name": entry.get("dataset_name", "unknown"),
                    "model_name": entry.get("model_name", "unknown"),
                    "fold": str(fold),
                    "best_seen_val": float(best_seen_val),
                    "final_val": float(final_val),
                    "gap": float(final_val - best_seen_val),
                    "relative_gap_pct": relative_gap_pct,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"No fold entries with metric '{metric}' found in {results_path}"
        )
    return df


def plot_gap_histograms(
    df: pd.DataFrame,
    output_path: Path,
    bins: int,
    ncols: int,
    dataset_filter: str | None,
    model_filter: str | None,
) -> None:
    if dataset_filter:
        df = df[df["dataset_name"].str.contains(dataset_filter, regex=True)]
    if model_filter:
        df = df[df["model_name"].str.contains(model_filter, regex=True)]

    if df.empty:
        raise ValueError("No rows left after applying filters")

    df = df.dropna(subset=["relative_gap_pct"])
    if df.empty:
        raise ValueError("All relative gaps are undefined after filtering")

    group_keys = sorted(
        df[["dataset_name", "model_name"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    plt.style.use("default")

    nrows = math.ceil(len(group_keys) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.4 * ncols, 3.8 * nrows),
        squeeze=False,
        sharex=True,
        sharey=False,
    )

    all_gaps = df["relative_gap_pct"].to_numpy()
    min_gap = float(np.min(all_gaps))
    max_gap = float(np.max(all_gaps))
    if np.isclose(min_gap, max_gap):
        min_gap -= 1e-6
        max_gap += 1e-6

    flat_axes = axes.flatten()
    for ax, (dataset_name, model_name) in zip(flat_axes, group_keys):
        subset = df[
            (df["dataset_name"] == dataset_name) & (df["model_name"] == model_name)
        ]

        ax.hist(
            subset["relative_gap_pct"],
            bins=bins,
            range=(min_gap, max_gap),
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.8,
        )
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{dataset_name}\n{model_name} (n={len(subset)})", fontsize=10)
        ax.set_xlabel("relative gap [%]")
        ax.set_ylabel("count")

        mean_gap = subset["relative_gap_pct"].mean()
        median_gap = subset["relative_gap_pct"].median()
        ax.text(
            0.98,
            0.95,
            f"mean={mean_gap:.2f}%\nmedian={median_gap:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    for ax in flat_axes[len(group_keys) :]:
        ax.axis("off")

    fig.suptitle(
        "Validation gap distribution by dataset and model\n"
        "relative gap [%] = 100 * (final validation loss - best validation loss) / |best validation loss|",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["dataset_name", "model_name"])["relative_gap_pct"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_index()
    )
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot histograms of final-vs-best validation loss gaps from results.json"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "experiments" / "results.json",
        help="Path to results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "figures"
        / "val_gap_histograms.png",
        help="Output image path",
    )
    parser.add_argument(
        "--metric",
        default="mse",
        help="Validation metric to compare against best_seen_val_loss",
    )
    parser.add_argument("--bins", type=int, default=25, help="Histogram bins")
    parser.add_argument("--ncols", type=int, default=4, help="Number of subplot columns")
    parser.add_argument(
        "--dataset-filter",
        default=None,
        help="Optional regex filter applied to dataset_name",
    )
    parser.add_argument(
        "--model-filter",
        default=None,
        help="Optional regex filter applied to model_name",
    )
    args = parser.parse_args()

    df = load_gap_dataframe(args.results, args.metric)
    print_summary(df)
    plot_gap_histograms(
        df=df,
        output_path=args.output,
        bins=args.bins,
        ncols=args.ncols,
        dataset_filter=args.dataset_filter,
        model_filter=args.model_filter,
    )
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
