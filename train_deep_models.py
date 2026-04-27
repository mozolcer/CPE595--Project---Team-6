import argparse
import csv
import json
import random
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from deep_vision import CLASSES, build_model, build_transform, resolve_model_name


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "trashnet-master" / "data" / "dataset-resized"
OUTPUT_DIR = BASE_DIR / "output" / "deep"


@dataclass(frozen=True)
class Experiment:
    name: str
    augment: str
    class_weights: bool
    sampler: bool
    freeze_epochs: int
    head_only: bool
    pretrained: bool
    image_size: int
    epochs: int
    lr: float
    weight_decay: float
    label_smoothing: float


class TrashNetDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image = Image.open(record["path"]).convert("RGB")
        return self.transform(image), record["target"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(device):
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_device_type(device):
    return "cuda" if device.type == "cuda" else "cpu"


def list_samples(data_dir, limit_per_class):
    samples = []
    for target, cls in enumerate(CLASSES):
        paths = sorted((data_dir / cls).glob("*.jpg"))
        if limit_per_class:
            paths = paths[:limit_per_class]
        for path in paths:
            samples.append({"path": str(path), "label": cls, "target": target})
    return samples


def make_splits(samples, seed):
    y = np.array([sample["target"] for sample in samples])
    idx = np.arange(len(samples))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(idx, y))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(sss2.split(temp_idx, y[temp_idx]))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

    return {
        "train": [samples[i] for i in train_idx],
        "val": [samples[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
    }


def save_splits(splits, path, args):
    payload = {
        "seed": args.seed,
        "limit_per_class": args.limit_per_class,
        "classes": CLASSES,
        "splits": splits,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def class_counts(records):
    counts = Counter(record["target"] for record in records)
    return {CLASSES[i]: counts[i] for i in range(len(CLASSES))}


def build_experiments(args):
    plans = {
        "smoke": [
            ("smoke", "light", True, False, 0, False, args.pretrained, 1),
        ],
        "core": [
            ("head_only", "light", True, False, args.epochs, True, args.pretrained, args.head_epochs),
            ("full_finetune", "light", True, False, args.freeze_epochs, False, args.pretrained, args.epochs),
        ],
        "full": [
            ("head_only", "light", True, False, args.epochs, True, args.pretrained, args.head_epochs),
            ("full_finetune", "light", True, False, args.freeze_epochs, False, args.pretrained, args.epochs),
            ("strong_aug", "strong", True, False, args.freeze_epochs, False, args.pretrained, args.epochs),
            ("weighted_sampler", "light", False, True, args.freeze_epochs, False, args.pretrained, args.epochs),
            ("no_class_weights", "light", False, False, args.freeze_epochs, False, args.pretrained, args.epochs),
        ],
        "scratch": [
            ("scratch_light_weighted_loss", "light", True, False, 0, False, False, args.epochs),
            ("scratch_strong_weighted_loss", "strong", True, False, 0, False, False, args.epochs),
            ("scratch_strong_weighted_sampler", "strong", False, True, 0, False, False, args.epochs),
        ],
        "scratch_light": [
            ("scratch_light_weighted_loss", "light", True, False, 0, False, False, args.epochs),
        ],
    }
    experiments = []
    for lr in args.learning_rates:
        for name, augment, class_weights, sampler, freeze_epochs, head_only, pretrained, epochs in plans[args.ablation_plan]:
            exp_name = name if len(args.learning_rates) == 1 else f"{name}_lr{lr:g}"
            experiments.append(Experiment(
                name=exp_name,
                augment=augment,
                class_weights=class_weights,
                sampler=sampler,
                freeze_epochs=freeze_epochs,
                head_only=head_only,
                pretrained=pretrained,
                image_size=args.image_size,
                epochs=epochs,
                lr=lr,
                weight_decay=args.weight_decay,
                label_smoothing=args.label_smoothing,
            ))
    return experiments


def set_backbone_trainable(model, trainable):
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    classifier = model.get_classifier()
    for parameter in classifier.parameters():
        parameter.requires_grad = True


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def build_loaders(splits, experiment, args, device):
    train_transform = build_transform(experiment.image_size, split="train", augment=experiment.augment)
    eval_transform = build_transform(experiment.image_size, split="eval", augment=experiment.augment)
    train_ds = TrashNetDataset(splits["train"], train_transform)
    val_ds = TrashNetDataset(splits["val"], eval_transform)
    test_ds = TrashNetDataset(splits["test"], eval_transform)

    sampler = None
    if experiment.sampler:
        targets = [record["target"] for record in splits["train"]]
        counts = np.bincount(targets, minlength=len(CLASSES))
        sample_weights = torch.DoubleTensor([1.0 / counts[target] for target in targets])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        shuffle=sampler is None,
        sampler=sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def build_criterion(records, experiment, device):
    if not experiment.class_weights:
        return torch.nn.CrossEntropyLoss(label_smoothing=experiment.label_smoothing)
    targets = [record["target"] for record in records]
    counts = np.bincount(targets, minlength=len(CLASSES))
    weights = counts.sum() / (len(CLASSES) * counts)
    return torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=experiment.label_smoothing,
    )


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=amp_device_type(device), enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = targets.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
    return total_loss / total_items


@torch.inference_mode()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_true = []
    all_pred = []
    all_prob = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=amp_device_type(device), enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
        batch_size = targets.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        all_true.extend(targets.cpu().tolist())
        all_pred.extend(predictions.cpu().tolist())
        all_prob.extend(probabilities.cpu().tolist())

    report = classification_report(
        all_true,
        all_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": total_loss / total_items,
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_true, all_pred, average="weighted", zero_division=0),
        "report": report,
        "y_true": all_true,
        "y_pred": all_pred,
        "probabilities": all_prob,
    }


@torch.inference_mode()
def measure_latency(model, loader, device, use_amp, warmup=10, repeats=100):
    model.eval()
    images, _ = next(iter(loader))
    sample = images[:1].to(device)
    for _ in range(warmup):
        with torch.amp.autocast(device_type=amp_device_type(device), enabled=use_amp):
            model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.amp.autocast(device_type=amp_device_type(device), enabled=use_amp):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def plot_confusion(metrics, path, title):
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(metrics, path, title):
    f1s = [metrics["report"][cls]["f1-score"] for cls in CLASSES]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(CLASSES, f1s, color=sns.color_palette("muted", len(CLASSES)))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    for bar, value in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_history(path, history):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def write_leaderboard(path, rows):
    fieldnames = [
        "model",
        "timm_model",
        "experiment",
        "best_epoch",
        "val_macro_f1",
        "test_macro_f1",
        "test_accuracy",
        "median_latency_ms",
        "params",
        "trainable_params",
        "seconds",
        "checkpoint",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["val_macro_f1"], reverse=True))


def write_summary(path, rows, splits, args):
    rows = sorted(rows, key=lambda row: row["val_macro_f1"], reverse=True)
    best = rows[0]
    lines = [
        "SortSmart Deep Vision Results",
        "=" * 50,
        "",
        f"Dataset: TrashNet ({len(CLASSES)} classes)",
        f"Split: {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test",
        f"Train class counts: {class_counts(splits['train'])}",
        f"Models: {', '.join(args.models)}",
        f"Ablation plan: {args.ablation_plan}",
        "",
        f"Best run: {best['model']} / {best['experiment']}",
        f"Validation Macro-F1: {best['val_macro_f1']:.4f}",
        f"Test Macro-F1: {best['test_macro_f1']:.4f}",
        f"Test Accuracy: {best['test_accuracy']:.4f}",
        f"Median latency: {best['median_latency_ms']:.2f} ms",
        f"Checkpoint: {best['checkpoint']}",
        "",
        "Leaderboard:",
    ]
    for row in rows:
        lines.append(
            f"  {row['model']:30s} {row['experiment']:20s} "
            f"val_f1={row['val_macro_f1']:.4f} test_f1={row['test_macro_f1']:.4f} "
            f"acc={row['test_accuracy']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n")


def train_run(model_key, experiment, splits, args, device):
    run_dir = args.output_dir / "runs" / model_key / experiment.name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = build_loaders(splits, experiment, args, device)
    model = build_model(model_key, pretrained=experiment.pretrained).to(device)
    if experiment.head_only or experiment.freeze_epochs > 0:
        set_backbone_trainable(model, False)
    params, trainable_params = count_parameters(model)
    if not experiment.head_only:
        trainable_params = params

    criterion = build_criterion(splits["train"], experiment, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment.lr, weight_decay=experiment.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=experiment.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(amp_device_type(device), enabled=use_amp)

    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    best_path = run_dir / "checkpoint_best.pt"

    for epoch in range(1, experiment.epochs + 1):
        if not experiment.head_only and epoch == experiment.freeze_epochs + 1:
            set_backbone_trainable(model, True)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)
        print(
            f"{model_key:30s} {experiment.name:18s} "
            f"epoch={epoch:02d}/{experiment.epochs:02d} "
            f"train_loss={train_loss:.4f} val_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_key": model_key,
                "timm_model": resolve_model_name(model_key),
                "classes": CLASSES,
                "image_size": experiment.image_size,
                "augment": experiment.augment,
                "experiment": asdict(experiment),
                "state_dict": model.state_dict(),
                "val_metrics": {
                    "loss": val_metrics["loss"],
                    "accuracy": val_metrics["accuracy"],
                    "macro_f1": val_metrics["macro_f1"],
                    "weighted_f1": val_metrics["weighted_f1"],
                },
                "best_epoch": best_epoch,
            }, best_path)
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            break

    write_history(run_dir / "history.csv", history)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, use_amp)
    median_latency = measure_latency(model, test_loader, device, use_amp, repeats=args.latency_repeats)

    plot_confusion(test_metrics, run_dir / "confusion_matrix.png", f"{model_key} / {experiment.name}")
    plot_per_class_f1(test_metrics, run_dir / "per_class_f1.png", f"{model_key} / {experiment.name}")

    metrics_payload = {
        "model": model_key,
        "timm_model": resolve_model_name(model_key),
        "experiment": asdict(experiment),
        "best_epoch": best_epoch,
        "params": params,
        "trainable_params": trainable_params,
        "val": checkpoint["val_metrics"],
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "report": test_metrics["report"],
        },
        "median_latency_ms": median_latency,
        "seconds": time.perf_counter() - started,
        "checkpoint": str(best_path),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    return {
        "model": model_key,
        "timm_model": resolve_model_name(model_key),
        "experiment": experiment.name,
        "best_epoch": best_epoch,
        "val_macro_f1": checkpoint["val_metrics"]["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "median_latency_ms": median_latency,
        "params": params,
        "trainable_params": trainable_params,
        "seconds": metrics_payload["seconds"],
        "checkpoint": str(best_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train deep vision models for SortSmart.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=[
        "efficientnetv2_s",
        "convnext_tiny",
        "swin_tiny_patch4_window7_224",
        "vit_base_patch16_224",
    ])
    parser.add_argument("--ablation-plan", choices=["smoke", "core", "full", "scratch", "scratch_light"], default="full")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[3e-4])
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = choose_device(args.device)
    print(f"Device: {device}")

    samples = list_samples(args.data_dir, args.limit_per_class)
    splits = make_splits(samples, args.seed)
    split_name = f"split_seed{args.seed}" + (f"_limit{args.limit_per_class}" if args.limit_per_class else "")
    save_splits(splits, args.output_dir / f"{split_name}.json", args)
    print(f"Split: {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test")
    print(f"Train counts: {class_counts(splits['train'])}")

    rows = []
    for model_key in args.models:
        for experiment in build_experiments(args):
            rows.append(train_run(model_key, experiment, splits, args, device))
            write_leaderboard(args.output_dir / "leaderboard.csv", rows)
            write_summary(args.output_dir / "deep_results_summary.txt", rows, splits, args)

    rows = sorted(rows, key=lambda row: row["val_macro_f1"], reverse=True)
    shutil.copyfile(rows[0]["checkpoint"], args.output_dir / "best_model.pt")
    write_leaderboard(args.output_dir / "leaderboard.csv", rows)
    write_summary(args.output_dir / "deep_results_summary.txt", rows, splits, args)
    print(f"Best checkpoint: {rows[0]['checkpoint']}")


if __name__ == "__main__":
    main()
