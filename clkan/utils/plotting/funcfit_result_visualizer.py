# This file was created mostly by ChatGPT 5.2 with small changes from us
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from clkan.utils.plotting.pandas_dataloader import load_json_make_pandas

# =========================
# CONFIGURATION
# =========================
CONFIG = {
    # dataframe column names
    "dataset_col": "dataset_name",
    "x_col": "num_grids",
    "layers_col": "layers",              # architecture definition (e.g. "64,64")
    "batchnorm_col": "use_norm",
    "grid_col": "extra_args.clifford_grid",
    "clifford_type_col": "extra_args.clifford_rbf",  # "naive" or "cliffordspace"

    # metrics to plot
    "metrics": {
        "mse": {"mean": "test_losses.mean.mse", "std": "test_losses.std.mse"},
        "mae": {"mean": "test_losses.mean.mae", "std": "test_losses.std.mae"},
    },

    # figure / style
    "figsize": (16, 4),
    "palette": "tab10",
    "alpha": 0.85,
    "dpi": 100,

    # marker configuration
    "marker_size": 50,     
    "x_jitter_frac": 0.45,      # horizontal jitter inside bins
    "sep_alpha": 0.8,          # vertical separator visibility
}

from matplotlib.lines import Line2D


def plot_experiment_results(df, CONFIG):
    """
    Plot CVKAN / CliffordKAN experiment results.

    One figure is created per (dataset, metric).

    Visual encoding after the requested change:
    --------------------------------------------
    * Marker SHAPE      -> RBF type (naive vs cliffordspace)
    * Marker FILL       -> architecture size (small vs large)
    * Marker COLOR      -> batchnorm usage

    Ordering inside each x-bin is kept identical to the previous implementation.
    """

    sns.set(style="whitegrid", context="paper", font_scale=1.3)

    # -------------------------
    # Column shortcuts
    # -------------------------
    DATASET = CONFIG["dataset_col"]
    XCOL = CONFIG["x_col"]
    LAYERS = CONFIG["layers_col"]
    BN = CONFIG["batchnorm_col"]
    GRID = CONFIG["grid_col"]
    CLIFFORD = CONFIG["clifford_type_col"]
    MODEL = "model_name"

    # Architecture size -> binary: small vs big (per dataset)
    def arch_is_big(layer):
        # smallest architecture is "small", everything else "big"
        return layer in ["1,2,1", "2,4,2,1"]

    for dataset in df[DATASET].unique():
        df_ds = df[df[DATASET] == dataset]

        # architecture definitions are dataset-specific
        layers_list = sorted(df_ds[LAYERS].unique())
        bn_list = sorted(df_ds[BN].unique())
        print(layers_list)
        print(bn_list)

        # ---------------------------------
        # Visual encodings
        # ---------------------------------
        # Color encodes BatchNorm usage
        palette = sns.color_palette(CONFIG["palette"], n_colors=len(bn_list))
        bn_color = dict(zip(bn_list, palette))

        # Shape now encodes RBF type
        rbf_marker = {
            "cliffordspace": "o",
            "naive": "_",
        }


        # Ordering inside bins (unchanged)
        clifford_order = ["cliffordspace", "naive"]

        for metric, cols in CONFIG["metrics"].items():
            mean_col = cols["mean"]
            std_col = cols["std"]

            fig, ax = plt.subplots(figsize=CONFIG["figsize"], dpi=CONFIG["dpi"])

            # =========================
            # X-axis bins
            # =========================
            x_bins = []
            x_labels = []

            # CVKANWrapper (single bin)
            x_bins.append(("CVKANWrapper",))
            x_labels.append("CVKAN")

            # CliffordKAN bins
            df_ck = df_ds[df_ds[MODEL] == "CliffordKAN"]

            x_bins.append(("CliffordKAN", "full_grid"))
            x_labels.append("F")

            #x_bins.append(("CliffordKAN", "independant_grid"))
            #x_labels.append("I")

            # random grid bins, one per grid count
            random_grids = sorted(
                df_ck[df_ck[GRID] == "random_grid"][XCOL].unique()
            )
            for ng in random_grids:
                if ng not in [2,3,4,6,8]:
                    continue
                x_bins.append(("CliffordKAN", "random_grid", ng))
                x_labels.append(f"S-{ng}")

            # =========================
            # Plot bins
            # =========================
            for x_idx, key in enumerate(x_bins):
                # Select rows belonging to this bin
                if key[0] == "CVKANWrapper":
                    g = df_ds[df_ds[MODEL] == "CVKANWrapper"]
                else:
                    model, grid = key[0], key[1]
                    g = df_ck[df_ck[GRID] == grid]
                    if len(key) == 3:
                        g = g[g[XCOL] == key[2]]

                if g.empty:
                    continue

                # Deterministic ordering inside each bin
                clifford_order_dict = {v: i for i, v in enumerate(clifford_order)}
                g = g.copy()
                g["_order"] = [
                    (
                        layers_list.index(r[LAYERS]),
                        bn_list.index(r[BN]),clifford_order_dict.get(r[CLIFFORD], 999),
                    )
                    for _, r in g.iterrows()
                ]
                g = g.sort_values("_order")

                # Horizontal jitter to avoid overlap
                n = len(g)
                offsets = (
                    np.linspace(-CONFIG["x_jitter_frac"], CONFIG["x_jitter_frac"], n)
                    if n > 1 else [0.0]
                )

                # -------------------------
                # Draw points
                # -------------------------
                big_arches = []
                small_arches = []
                last_was_small = -1
                for offset, (_, row) in zip(offsets, g.iterrows()):
                    x = x_idx + offset

                    color = bn_color[row[BN]]
                    marker = rbf_marker.get(row[CLIFFORD], "_")
                    size = CONFIG["marker_size"]

                    # Filled vs hollow now encodes architecture size
                    # (simple split: smallest architecture = hollow, larger = filled)
                    is_large_arch = arch_is_big(row[LAYERS])
                    if last_was_small == -1 and not is_large_arch:
                        last_was_small = True
                    elif last_was_small == -1 and is_large_arch:
                        last_was_small = False
                    if last_was_small and is_large_arch:
                        plt.axvspan(small_arches[0]-0.04, (small_arches[-1] + x)/2,  color="grey", alpha=0.4)
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
                        row[mean_col],
                        yerr=row[std_col],
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
            ax.set_xlabel("Model / Grid strategy")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{dataset[3:]}")

            # Vertical separators between bins
            for i in range(len(x_bins) - 1):
                ax.axvline(
                    i + 0.5,
                    color="black",
                    linewidth=2.5,
                    alpha=CONFIG["sep_alpha"],
                    zorder=0,
                )

            # =========================
            # Legends
            # =========================
            # BatchNorm legend (color)
            bn_legend = [
                Line2D(
                    [0], [0], marker="o", color="none",
                    markerfacecolor=bn_color[b], markeredgecolor=bn_color[b],
                    label=b, markersize=10
                )
                for b in bn_list
            ]

            # RBF type legend (shape)
            rbf_legend = [
                Line2D([0], [0], marker=m, color="black",
                       linestyle="none", label=k, markersize=10)
                for k, m in rbf_marker.items()
            ]

            # Architecture size legend (fill)
            arch_legend = []
            for layer in layers_list:
                is_large_arch = layer != layers_list[0]
                arch_legend.append(
                        Line2D(
                            [0], [0],
                            marker="o",
                            color="black",
                            linestyle="none",
                            markerfacecolor="black" if is_large_arch else "none",
                            markeredgecolor="black",
                            label=layer,
                            markersize=10,
                            )
                        )
            if dataset == "ff_square":
                leg1 = ax.legend(handles=bn_legend, title="BatchNorm", loc="upper left", labelspacing=0.2)#,bbox_to_anchor=(0.4, 0))
                ax.add_artist(leg1)
                leg2 = ax.legend(handles=rbf_legend, title="RBF type", loc="upper right", labelspacing=0.2)#,bbox_to_anchor=(0.65, 0))
                ax.add_artist(leg2)
            #ax.legend(handles=arch_legend, title="Architecture size", loc="lower center", labelspacing=0.2)

            ax.set_yscale("log")#, linthresh=1)
            ax.set_xlim(-0.5,6.5)
            plt.tight_layout()
            plt.savefig(f"{metric}-{dataset}.svg", bbox_inches="tight")
            #plt.show()


if __name__ == "__main__":
    results_df = load_json_make_pandas(Path(__file__).parent.parent.parent / "experiments/results.json", filter_option="funcfit")
    print(len(results_df))
    plot_experiment_results(results_df,CONFIG)
    
