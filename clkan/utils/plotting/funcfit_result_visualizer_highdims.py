# This file was created mostly by ChatGPT 5.2 with small changes from us
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from clkan.utils.plotting.pandas_dataloader import load_json_make_pandas

# =========================
# CONFIGURATION
# =========================
CONFIG = {
    # dataframe column names
    "dataset_col": "dataset_name",
    "x_col": "num_grids",
    "layers_col": "layers",              # architecture definition (e.g. "64,64")
    "signature_col": "metric",              # NEW: categorical metric dimension (4 values)
    "grid_col": "extra_args.clifford_grid",

    # loss values
    "metrics": {
        "mse": {"mean": "test_losses.mean.mse", "std": "test_losses.std.mse"},
    },

    # figure / style
    "figsize": (16, 4),
    "palette": "tab10",
    "alpha": 0.85,
    "dpi": 100,

    # marker configuration
    "marker_shapes": ["o", "s", "_"],  # dataset encoding (up to 3 datasets)
    "marker_size": 100,
    "x_jitter_frac": 0.35,
    "sep_alpha": 0.8,
}


def plot_experiment_results(df, CONFIG):
    """
    Plot CliffordKAN experiment results across multiple datasets in a single figure.

    Fixed dimensions (removed):
    ---------------------------
    * RBF type (always cliffordspace)
    * Normalization (always node-wise)
    * CVKANWrapper (no longer exists)

    Visual encoding:
    ----------------
    * X-axis bin     -> grid strategy / number of grids
    * Marker SHAPE   -> dataset
    * Marker COLOR   -> metric (categorical, 4 values)
    * Marker FILL    -> architecture size

    Ordering inside each x-bin:
    ---------------------------
    1. dataset
    2. metric
    """

    sns.set(style="whitegrid", context="paper", font_scale=1.2)

    # -------------------------
    # Column shortcuts
    # -------------------------
    DATASET = CONFIG["dataset_col"]
    XCOL = CONFIG["x_col"]
    LAYERS = CONFIG["layers_col"]
    SIGNATURE = CONFIG["signature_col"]
    GRID = CONFIG["grid_col"]

    datasets = sorted(df[DATASET].unique())
    signatures = sorted(df[SIGNATURE].unique())
    #layers_list = sorted(df[LAYERS].unique())
    layers_list = ["1,1", "2,1,1", "2,2,1", "1,2,1", "2,4,2,1"]

    # ---------------------------------
    # Visual encodings
    # ---------------------------------
    # Dataset -> marker shape
    # Dataset -> marker shape (use highly distinguishable shapes)
    dataset_marker = {
        ds: m for ds, m in zip(datasets, ["o", "^", "_"])
    }

    # Metric -> color (same role BN had before)
    signatures = sorted(df[SIGNATURE].unique())
    palette = sns.color_palette(CONFIG["palette"], n_colors=len(signatures))
    signature_color = dict(zip(signatures, palette))

    # Architecture size -> binary: small vs big (per dataset)
    def arch_is_big(layer):
        # smallest architecture is "small", everything else "big"
        return layer in ["1,2,1", "2,4,2,1"]

    # =========================
    # X-axis bins (CliffordKAN only)
    # =========================
    x_bins = []
    x_labels = []

    x_bins.append(("full_grid",))
    x_labels.append("F")

    #x_bins.append(("independant_grid",))
    #x_labels.append("I")

    # random grid bins (make sure they are included even if mixed with other grids)
    random_grids = sorted(
        df[df[GRID] == "random_grid"][XCOL].dropna().unique()
    )
    for ng in random_grids:
        if ng not in [2,3,4,6,8]:
            continue
        x_bins.append(("random", ng))
        x_labels.append(f"S-{ng}")

    # =========================
    # Plot
    # =========================
    fig, ax = plt.subplots(figsize=CONFIG["figsize"], dpi=CONFIG["dpi"])

    for x_idx, key in enumerate(x_bins):
        if key[0] == "random":
            g = df[(df[GRID] == "random_grid") & (df[XCOL] == key[1])]
        else:
            g = df[df[GRID] == key[0]]

        if g.empty:
            continue

        # Deterministic ordering inside bins
        g = g.copy()
        g["_order"] = [
            (
                arch_is_big(r[LAYERS]),
                #layers_list.index(r[LAYERS]),
                datasets.index(r[DATASET]),
                signatures.index(r[SIGNATURE]),
            )
            for _, r in g.iterrows()
        ]
        g = g.sort_values("_order")

        # Horizontal jitter
        n = len(g)
        offsets = (
            np.linspace(-CONFIG["x_jitter_frac"], CONFIG["x_jitter_frac"], n)
            if n > 1 else [0.0]
        )

        big_arches = []
        small_arches = []
        last_was_small = -1
        for offset, (_, row) in zip(offsets, g.iterrows()):
            x = x_idx + offset

            marker = dataset_marker[row[DATASET]]
            color = signature_color[row[SIGNATURE]]
            size = CONFIG["marker_size"]

            # Smallest architecture hollow, others filled
            is_large_arch = arch_is_big(row[LAYERS])
            if last_was_small == -1 and not is_large_arch:
                last_was_small = True
            elif last_was_small == -1 and is_large_arch:
                last_was_small = False
            if last_was_small and is_large_arch:
                plt.axvspan(small_arches[0]-0.135, (small_arches[-1] + x)/2,  color="grey", alpha=0.4)
                small_arches = []
            if is_large_arch:
                big_arches.append(x)
                last_was_small = False
            else:
                small_arches.append(x)
                last_was_small = True


            mfc = color# if is_large_arch else "none"
            mec = color

            ax.errorbar(
                x,
                row[CONFIG["metrics"]["mse"]["mean"]],
                yerr=row[CONFIG["metrics"]["mse"]["std"]],
                fmt=marker,
                linestyle="none",
                markersize=np.sqrt(size),
                mfc=mfc,
                mec=mec,
                ecolor=color,
                alpha=CONFIG["alpha"],
                capsize=3,
            )

    # =========================
    # Axis & separators
    # =========================
    ax.set_xticks(range(len(x_bins)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Grid strategy")
    ax.set_ylabel("MSE")

    for i in range(len(x_bins) - 1):
        ax.axvline(i + 0.5, color="black", linewidth=2.5, alpha=CONFIG["sep_alpha"], zorder=0)

    # =========================
    # Legends
    # =========================
    # Dataset legend (shape)
    dataset_legend = [
        Line2D([0], [0], marker=dataset_marker[d], color="black",
               linestyle="none", label=d, markersize=10)
        for d in datasets
    ]

    # Signature legend (color)
    signature_legend = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=signature_color[s], markeredgecolor=signature_color[s],
               label=s, markersize=10)
        for s in signatures
    ]

    # Architecture legend (binary size)
    arch_legend = [
        Line2D([0], [0], marker="o", color="black", linestyle="none",
               markerfacecolor="none", markeredgecolor="black",
               label="small", markersize=10),
        Line2D([0], [0], marker="o", color="black", linestyle="none",
               markerfacecolor="black", markeredgecolor="black",
               label="big", markersize=10),
    ]
    plt.tight_layout()
    ax.set_xlim(-0.5,5.5)
    zoomed = False
    if not zoomed:
        leg1 = fig.legend(handles=dataset_legend, title="Dataset", loc="upper left",labelspacing=0.2, ncol=3,bbox_to_anchor=(0.75,1.1))
        ax.add_artist(leg1)
        leg2 = fig.legend(handles=signature_legend, title="Signature", loc="upper right",labelspacing=0.2, ncol=4, bbox_to_anchor=(0.385,1.1))
        ax.add_artist(leg2)
        #ax.legend(handles=arch_legend, title="Architecture", loc="upper left",labelspacing=0.2 ,bbox_to_anchor=(0.75,1.15))
        ax.set_yscale("log")
        plt.savefig(f"funcfit_highdims.svg", bbox_inches="tight")
    else:
        ax.set_ylim(-1,4)
        plt.savefig(f"funcfit_highdims_zoomed.svg", bbox_inches="tight")
    #plt.show()



if __name__ == "__main__":
    results_df = load_json_make_pandas(Path(__file__).parent.parent.parent / "experiments/results.json",
                                       filter_option="highdims")
    print(len(results_df))
    plot_experiment_results(results_df, CONFIG)

