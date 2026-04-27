import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("output/deep/presentable")
OUT.mkdir(parents=True, exist_ok=True)

CLASSICAL_ROWS = [
    {
        "family": "Classical ML",
        "model": "Linear SVM",
        "experiment": "HOG+HSV",
        "val_macro_f1": 0.5707,
        "test_macro_f1": "",
        "test_accuracy": "",
        "median_latency_ms": "",
        "params": "",
        "checkpoint": "",
    },
    {
        "family": "Classical ML",
        "model": "Logistic Regression",
        "experiment": "HOG+HSV",
        "val_macro_f1": 0.6340,
        "test_macro_f1": "",
        "test_accuracy": "",
        "median_latency_ms": "",
        "params": "",
        "checkpoint": "",
    },
    {
        "family": "Classical ML",
        "model": "Random Forest",
        "experiment": "HOG+HSV",
        "val_macro_f1": 0.7323,
        "test_macro_f1": "",
        "test_accuracy": "",
        "median_latency_ms": "",
        "params": "",
        "checkpoint": "",
    },
    {
        "family": "Classical ML",
        "model": "RBF SVM",
        "experiment": "HOG+HSV",
        "val_macro_f1": 0.7060,
        "test_macro_f1": "",
        "test_accuracy": "",
        "median_latency_ms": "",
        "params": "",
        "checkpoint": "",
    },
    {
        "family": "Classical ML",
        "model": "Gradient Boosting",
        "experiment": "HOG+HSV",
        "val_macro_f1": 0.7847,
        "test_macro_f1": 0.7833,
        "test_accuracy": 0.7895,
        "median_latency_ms": 7.48,
        "params": "",
        "checkpoint": "output/best_model.joblib",
    },
]


def read_leaderboard(path, family):
    rows = []
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            rows.append({
                "family": family,
                "model": row["model"],
                "experiment": row["experiment"],
                "val_macro_f1": float(row["val_macro_f1"]),
                "test_macro_f1": float(row["test_macro_f1"]),
                "test_accuracy": float(row["test_accuracy"]),
                "median_latency_ms": float(row["median_latency_ms"]),
                "params": int(row["params"]),
                "checkpoint": row["checkpoint"],
            })
    return rows


def suffix_experiment(rows, suffix):
    for row in rows:
        row["experiment"] = f"{row['experiment']}_{suffix}"
    return rows


def fmt(value):
    if value == "":
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path, rows):
    fields = [
        "family", "model", "experiment", "val_macro_f1", "test_macro_f1",
        "test_accuracy", "median_latency_ms", "params", "checkpoint",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    fields = ["family", "model", "experiment", "val_macro_f1", "test_macro_f1", "test_accuracy", "median_latency_ms"]
    lines = [
        "# SortSmart Combined Model Results",
        "",
        "| Family | Model | Experiment | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Latency ms |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[field]) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def plot_validation(rows, path):
    rows = sorted(rows, key=lambda row: float(row["val_macro_f1"]), reverse=True)
    labels = [f"{row['model']}\\n{row['experiment']}" for row in rows]
    values = [float(row["val_macro_f1"]) for row in rows]
    colors = [{"Classical ML": "#4C78A8", "Custom CNN": "#F58518", "Pretrained Deep": "#54A24B"}[row["family"]] for row in rows]
    height = max(7, 0.42 * len(rows))
    fig, ax = plt.subplots(figsize=(12, height))
    ax.barh(np.arange(len(rows)), values, color=colors)
    ax.set_yticks(np.arange(len(rows)), labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Validation Macro-F1")
    ax.set_title("All Trained Models: Validation Macro-F1")
    for i, value in enumerate(values):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_test_available(rows, path):
    rows = [row for row in rows if row["test_macro_f1"] != ""]
    rows = sorted(rows, key=lambda row: float(row["test_macro_f1"]), reverse=True)
    labels = [f"{row['model']}\\n{row['experiment']}" for row in rows]
    values = [float(row["test_macro_f1"]) for row in rows]
    colors = [{"Classical ML": "#4C78A8", "Custom CNN": "#F58518", "Pretrained Deep": "#54A24B"}[row["family"]] for row in rows]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * len(rows))))
    ax.barh(np.arange(len(rows)), values, color=colors)
    ax.set_yticks(np.arange(len(rows)), labels=labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Test Macro-F1")
    ax.set_title("Models With Test Results: Test Macro-F1")
    for i, value in enumerate(values):
        ax.text(value + 0.01, i, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_family_best(rows, path):
    best = {}
    for row in rows:
        if row["test_macro_f1"] == "":
            continue
        family = row["family"]
        if family not in best or row["test_macro_f1"] > best[family]["test_macro_f1"]:
            best[family] = row
    rows = [best[key] for key in ["Classical ML", "Custom CNN", "Pretrained Deep"] if key in best]
    labels = [row["family"] for row in rows]
    values = [float(row["test_macro_f1"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"][:len(rows)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Best Model By Family")
    for bar, row, value in zip(bars, rows, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}\\n{row['model']}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_neural_curves(path):
    series = [
        ("custom_cnn_scratch", "Scratch CNN", Path("output/deep/custom_cnn_scratch_light_ls_70ep/runs/custom_cnn_scratch/scratch_light_weighted_loss/history.csv")),
        ("convnext_tiny", "ConvNeXt-Tiny", Path("output/deep/pretrained_core_lr1e4_5ep/runs/convnext_tiny/full_finetune/history.csv")),
        ("swin_tiny_patch4_window7_224", "Swin-Tiny", Path("output/deep/pretrained_core_lr1e4_5ep/runs/swin_tiny_patch4_window7_224/full_finetune/history.csv")),
        ("vit_base_patch16_224", "ViT-B/16", Path("output/deep/pretrained_core_lr1e4_5ep/runs/vit_base_patch16_224/full_finetune/history.csv")),
        ("efficientnetv2_s", "EfficientNetV2-S", Path("output/deep/pretrained_core_lr1e4_5ep/runs/efficientnetv2_s/full_finetune/history.csv")),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    for _, label, hist_path in series:
        rows = list(csv.DictReader(hist_path.open()))
        ax.plot(
            [int(row["epoch"]) for row in rows],
            [float(row["val_macro_f1"]) for row in rows],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Neural Model Training Curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    custom = suffix_experiment(
        read_leaderboard("output/deep/custom_cnn_scratch_35ep/leaderboard.csv", "Custom CNN"),
        "35ep",
    )
    improved = suffix_experiment(
        read_leaderboard("output/deep/custom_cnn_scratch_light_ls_70ep/leaderboard.csv", "Custom CNN"),
        "70ep_ls0.05",
    )
    pretrained = suffix_experiment(
        read_leaderboard("output/deep/pretrained_core_lr1e4_5ep/leaderboard.csv", "Pretrained Deep"),
        "5ep_lr1e-4",
    )
    rows = CLASSICAL_ROWS + custom + improved + pretrained
    rows = sorted(rows, key=lambda row: float(row["val_macro_f1"]), reverse=True)
    write_csv(OUT / "combined_all_model_results.csv", rows)
    write_markdown(OUT / "combined_all_model_results.md", rows)
    plot_validation(rows, OUT / "combined_validation_macro_f1_all_models.png")
    plot_test_available(rows, OUT / "combined_test_macro_f1_available_models.png")
    plot_family_best(rows, OUT / "combined_best_by_family_test_macro_f1.png")
    plot_neural_curves(OUT / "combined_neural_training_curves.png")


if __name__ == "__main__":
    main()
