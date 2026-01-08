import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ======================================================
# Directories
# ======================================================
RUNS_DIR = "runs"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ======================================================
# Load experiments
# ======================================================
def load_experiments(runs_dir, pattern=None):
    data = {}

    for root, _, files in os.walk(runs_dir):
        if "metrics.csv" in files:
            if pattern and pattern not in root:
                continue

            exp_name = root.replace(runs_dir + "/", "")
            csv_path = os.path.join(root, "metrics.csv")

            try:
                data[exp_name] = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[WARNING] Failed to load {csv_path}: {e}")

    if len(data) == 0:
        raise RuntimeError(
            f"No metrics.csv found under {runs_dir} (pattern={pattern})"
        )

    print(f"Loaded {len(data)} experiments for comparison.")
    return data

# ======================================================
# Helper: metric curve
# ======================================================
def plot_metric(data, metric, ylabel, filename):
    plt.figure(figsize=(7, 5))

    for name, df in data.items():
        if metric not in df.columns:
            continue
        plt.plot(df["epoch"], df[metric], label=name)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()

# ======================================================
# Helper: alignment–uniformity trajectory
# ======================================================
def plot_alignment_uniformity_trajectory(data, filename):
    plt.figure(figsize=(7, 5))

    for name, df in data.items():
        if "alignment" not in df.columns or "uniformity" not in df.columns:
            continue

        x = df["alignment"].values
        y = df["uniformity"].values

        line, = plt.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=1,
            label=name,
        )

        color = line.get_color()

        # Start point
        plt.scatter(
            x[0], y[0],
            color=color,
            marker="s",
            s=60,
            zorder=3,
        )

        # End point
        plt.scatter(
            x[-1], y[-1],
            color=color,
            marker="*",
            s=100,
            zorder=3,
        )

    plt.xlabel("Alignment (lower is better)")
    plt.ylabel("Uniformity (lower is better)")
    plt.grid(True)

    legend_elements = [
        Line2D([0], [0], marker="s", color="black",
               linestyle="None", markersize=8, label="Start"),
        Line2D([0], [0], marker="*", color="black",
               linestyle="None", markersize=10, label="End"),
    ]

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles + legend_elements,
        labels + ["Start", "End"],
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()

# ======================================================
# Helper: final value bar plot (linear probe / kNN)
# ======================================================
def plot_final_metric(data, metric, ylabel, filename):
    names = []
    values = []

    for name, df in data.items():
        if metric not in df.columns:
            continue
        names.append(name)
        values.append(df[metric].iloc[-1])

    if len(values) == 0:
        print(f"[WARNING] No data for {metric}")
        return

    plt.figure(figsize=(8, 4))
    plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=300)
    plt.close()

# ======================================================
# Main
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional substring to filter run paths"
    )
    args = parser.parse_args()

    data = load_experiments(args.runs_dir, pattern=args.pattern)

    # ---------- Core plots ----------
    plot_metric(
        data,
        metric="val_acc",
        ylabel="Validation Accuracy",
        filename="accuracy_comparison.png",
    )

    plot_metric(
        data,
        metric="alignment",
        ylabel="Alignment (lower is better)",
        filename="alignment_comparison.png",
    )

    plot_metric(
        data,
        metric="uniformity",
        ylabel="Uniformity (lower is better)",
        filename="uniformity_comparison.png",
    )

    plot_alignment_uniformity_trajectory(
        data,
        filename="alignment_uniformity_trajectory.png",
    )

    # ---------- Extended evaluation ----------
    plot_final_metric(
        data,
        metric="linear_probe_acc",
        ylabel="Linear Probe Accuracy",
        filename="linear_probe_comparison.png",
    )

    plot_final_metric(
        data,
        metric="knn_acc",
        ylabel="k-NN Accuracy",
        filename="knn_comparison.png",
    )

    print("All comparison figures saved to ./figures/")

# ======================================================
if __name__ == "__main__":
    main()
