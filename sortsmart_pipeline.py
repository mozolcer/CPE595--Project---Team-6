"""
SortSmart - Feature Extraction, Training, and Evaluation Pipeline
CPE 595 Applied Machine Learning - Team 6
"""

import time
import warnings
import numpy as np
import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
)
from skimage.feature import hog

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "trashnet-master" / "data" / "dataset-resized"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (128, 128)
SEED = 42
AUG_TARGET = 500


# ── Feature extraction ────────────────────────────────────────────────

def extract_features(img_bgr):
    """HOG (8x8 ppc) + L1-normalized HSV color histogram."""
    resized = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm="L2-Hys",
                   feature_vector=True)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = np.concatenate([
        cv2.calcHist([hsv], [ch], None, [32],
                     [0, 180] if ch == 0 else [0, 256]).flatten()
        for ch in range(3)
    ])
    hist = hist / (hist.sum() + 1e-7)

    return np.concatenate([hog_feat, hist])


def augment_image(img):
    """Random flip + brightness jitter + small rotation."""
    rng = np.random.default_rng()
    aug = img.copy()
    if rng.random() > 0.5:
        aug = cv2.flip(aug, 1)
    aug = np.clip(aug.astype(np.int16) + rng.integers(-40, 41), 0, 255).astype(np.uint8)
    angle = rng.uniform(-15, 15)
    h, w = aug.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return aug


# ── Data loading ──────────────────────────────────────────────────────

def load_raw_images():
    """Load raw images grouped by class. No feature extraction yet."""
    data = {}
    for cls in CLASSES:
        imgs = []
        for p in sorted((DATA_DIR / cls).glob("*.jpg")):
            img = cv2.imread(str(p))
            if img is not None:
                imgs.append(img)
        data[cls] = imgs
        print(f"  {cls:>10s}: {len(imgs)} images")
    return data


def images_to_features(images, labels):
    """Convert list of BGR images to feature matrix."""
    X = np.array([extract_features(img) for img in images])
    y = np.array(labels)
    return X, y


def augment_training_set(images, labels):
    """Oversample minority classes with augmentation up to AUG_TARGET."""
    aug_images, aug_labels = list(images), list(labels)
    np.random.seed(SEED)

    for cls in CLASSES:
        cls_imgs = [img for img, lbl in zip(images, labels) if lbl == cls]
        n_aug = max(0, AUG_TARGET - len(cls_imgs))
        if n_aug > 0:
            for i in range(n_aug):
                aug_images.append(augment_image(cls_imgs[i % len(cls_imgs)]))
                aug_labels.append(cls)
            print(f"    {cls:>10s}: +{n_aug} augmented -> {len(cls_imgs) + n_aug}")

    return aug_images, aug_labels


# ── Splitting ─────────────────────────────────────────────────────────

def stratified_split(images, labels):
    """70/15/15 stratified split. Returns image lists + label lists."""
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    idx = np.arange(len(labels))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(sss1.split(idx, y_enc))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_rel, test_rel = next(sss2.split(temp_idx, y_enc[temp_idx]))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

    def gather(idxs):
        return [images[i] for i in idxs], [labels[i] for i in idxs]

    return gather(train_idx), gather(val_idx), gather(test_idx), le


# ── Training / evaluation ─────────────────────────────────────────────

def train_and_eval(name, clf, X_tr, y_tr, X_val, y_val):
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    mf1 = f1_score(y_val, y_pred, average="macro")
    print(f"  {name:30s}  acc={acc:.4f}  F1={mf1:.4f}  ({elapsed:.1f}s)")
    return clf, mf1


def plot_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_f1_bars(report_dict, labels, title, path):
    f1s = [report_dict[c]["f1-score"] for c in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, f1s, color=sns.color_palette("muted", len(labels)))
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 Score"); ax.set_title(title)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SortSmart ML Pipeline")
    print("=" * 60)

    # Load raw images
    print("\n[1/5] Loading images...")
    raw_data = load_raw_images()
    all_images = [img for cls in CLASSES for img in raw_data[cls]]
    all_labels = [cls for cls in CLASSES for _ in raw_data[cls]]

    # Split BEFORE augmentation (no data leakage)
    print("\n[2/5] Stratified split (70/15/15)...")
    (tr_imgs, tr_lbls), (val_imgs, val_lbls), (te_imgs, te_lbls), le = \
        stratified_split(all_images, all_labels)
    print(f"  Train: {len(tr_lbls)}  Val: {len(val_lbls)}  Test: {len(te_lbls)}")

    # Augment training set only
    print("\n[3/5] Augmenting training set (minority classes)...")
    tr_imgs, tr_lbls = augment_training_set(tr_imgs, tr_lbls)
    print(f"  Training set after augmentation: {len(tr_lbls)}")

    # Extract features
    print("\n[4/5] Extracting features (HOG + HSV)...")
    X_tr, y_tr = images_to_features(tr_imgs, tr_lbls)
    X_val, y_val = images_to_features(val_imgs, val_lbls)
    X_te, y_te = images_to_features(te_imgs, te_lbls)
    print(f"  Feature dim: {X_tr.shape[1]}")

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_te = scaler.transform(X_te)

    # Train models
    print("\n[5/5] Training & evaluating models...")
    models = {
        "Linear SVM":          LinearSVC(C=1.0, max_iter=5000, random_state=SEED),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=500,
                                                      random_state=SEED, n_jobs=-1),
        "RBF SVM":             SVC(C=10, gamma="scale", random_state=SEED),
        "Gradient Boosting":   HistGradientBoostingClassifier(max_iter=300,
                                                              max_depth=8,
                                                              random_state=SEED),
    }

    results = {}
    for name, clf in models.items():
        fitted, f1 = train_and_eval(name, clf, X_tr, y_tr, X_val, y_val)
        results[name] = (fitted, f1)

    best_name = max(results, key=lambda k: results[k][1])
    best_clf = results[best_name][0]
    print(f"\n  Best: {best_name} (val F1={results[best_name][1]:.4f})")

    # Calibrate for probability output
    if not hasattr(best_clf, "predict_proba"):
        best_clf = CalibratedClassifierCV(best_clf, cv=3, method="sigmoid")
        best_clf.fit(X_tr, y_tr)

    # Final test evaluation
    print("\n--- Test set results ---")
    y_pred = best_clf.predict(X_te)
    test_acc = accuracy_score(y_te, y_pred)
    test_f1 = f1_score(y_te, y_pred, average="macro")
    report = classification_report(y_te, y_pred, target_names=CLASSES, output_dict=True)
    report_str = classification_report(y_te, y_pred, target_names=CLASSES)
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  Macro-F1 : {test_f1:.4f}\n")
    print(report_str)

    # Latency
    times = [0] * 100
    sample = X_te[:1]
    for i in range(100):
        t0 = time.time(); best_clf.predict(sample); times[i] = time.time() - t0
    med_ms = np.median(times) * 1000
    print(f"  Median inference latency: {med_ms:.2f} ms")

    # Save artifacts
    plot_confusion(y_te, y_pred, CLASSES,
                   f"Confusion Matrix - {best_name} (Test)", OUTPUT_DIR / "confusion_matrix.png")
    plot_f1_bars(report, CLASSES,
                 f"Per-Class F1 - {best_name} (Test)", OUTPUT_DIR / "per_class_f1.png")
    joblib.dump(best_clf, OUTPUT_DIR / "best_model.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")
    joblib.dump(le, OUTPUT_DIR / "label_encoder.joblib")

    with open(OUTPUT_DIR / "results_summary.txt", "w") as f:
        f.write("SortSmart - Midterm Results Summary\n")
        f.write("=" * 50 + "\n\n")
        total_orig = sum(len(raw_data[c]) for c in CLASSES)
        f.write(f"Dataset: TrashNet (6 classes, {total_orig} images)\n")
        f.write(f"Features: HOG (9 orient, 8x8 ppc) + HSV histogram (32 bins x 3ch)\n")
        f.write(f"Feature dim: {X_tr.shape[1]}\n")
        f.write(f"Split: {len(tr_lbls)} train (aug) / {len(val_lbls)} val / {len(te_lbls)} test\n\n")
        f.write("Validation Macro-F1:\n")
        for name, (_, mf1) in results.items():
            marker = " <<<" if name == best_name else ""
            f.write(f"  {name:30s}: {mf1:.4f}{marker}\n")
        f.write(f"\nBest model: {best_name}\n")
        f.write(f"Test Accuracy : {test_acc:.4f}\n")
        f.write(f"Test Macro-F1 : {test_f1:.4f}\n")
        f.write(f"Median latency: {med_ms:.2f} ms\n\n")
        f.write("Per-class report (test):\n")
        f.write(report_str + "\n")

    print(f"\n  Artifacts saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
