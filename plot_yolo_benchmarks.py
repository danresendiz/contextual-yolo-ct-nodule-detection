import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt


def generate_benchmark_plots(data_dir: str, max_epochs: int = 100):
    plt.style.use("default")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    found_files, labels = [], []
    for version in [8, 9, 10, 11, 12]:
        pattern = os.path.join(data_dir, f"*olo*{version}*.csv")
        matches = glob.glob(pattern, recursive=False)
        if matches:
            found_files.append(matches[0])
            labels.append(f"YOLOv{version}")

    metrics = {
        "Precision": {"title": "A) Precision", "filename": "A_Precision.png"},
        "Recall": {"title": "B) Recall", "filename": "B_Recall.png"},
        "mAP50": {"title": "C) mAP50", "filename": "C_mAP50.png"},
        "mAP50-95": {"title": "D) mAP50-95", "filename": "D_mAP50-95.png"},
        "Train Box Loss": {"title": "E) Training Box Loss", "filename": "E_TrainLoss.png"},
        "Validation Box Loss": {"title": "F) Validation Box Loss", "filename": "F_ValLoss.png"},
    }

    model_dfs = []
    for file in found_files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        model_dfs.append(df.head(max_epochs))

    for metric_name, info in metrics.items():
        plt.figure(figsize=(10, 6))
        plt.gca().set_facecolor("white")

        for i, df in enumerate(model_dfs):
            epochs = df["epoch"] if "epoch" in df.columns else df.index
            col_name = None
            for col in df.columns:
                c = col.lower().replace(" ", "")
                if "(m)" in c:
                    continue
                if metric_name == "Precision" and ("precision" in c or "box_p" in c):
                    col_name = col
                    break
                elif metric_name == "Recall" and ("recall" in c or "box_r" in c):
                    col_name = col
                    break
                elif metric_name == "mAP50" and ("map50" in c or "map_0.5" in c) and "95" not in c:
                    col_name = col
                    break
                elif metric_name == "mAP50-95" and ("map50-95" in c or "0.5:0.95" in c):
                    col_name = col
                    break
                elif metric_name == "Train Box Loss" and "train/box_loss" in c:
                    col_name = col
                    break
                elif metric_name == "Validation Box Loss" and "val/box_loss" in c:
                    col_name = col
                    break

            if col_name:
                plt.plot(epochs, df[col_name], label=labels[i], color=colors[i], linewidth=2.5)

        plt.title(info["title"], fontsize=16, fontweight="bold", pad=15)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.grid(color="#e0e0e0", linestyle="--", linewidth=1)
        plt.xlim(0, max_epochs)

        legend_loc = "upper right" if "Loss" in metric_name else "lower right"
        legend = plt.legend(loc=legend_loc, frameon=True, fontsize=11)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#cccccc")

        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, info["filename"]), dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./results")
    args = parser.parse_args()
    generate_benchmark_plots(args.data_dir)