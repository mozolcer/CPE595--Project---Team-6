"""SortSmart Demo Web App — single-page Flask app."""

import numpy as np
import cv2
import joblib
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from skimage.feature import hog

app = Flask(__name__)

MODEL = joblib.load("output/best_model.joblib")
SCALER = joblib.load("output/scaler.joblib")

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (128, 128)
SAMPLES_DIR = Path("static/samples")

DISPOSAL = {
    "cardboard": ("Recycle", "Blue bin — flatten boxes first"),
    "glass":     ("Recycle", "Blue bin — rinse, remove lids"),
    "metal":     ("Recycle", "Blue bin — rinse cans"),
    "paper":     ("Recycle", "Blue bin — keep dry"),
    "plastic":   ("Recycle", "Blue bin — check local #s accepted"),
    "trash":     ("Trash",   "Black bin — general waste"),
}


def extract_features(img_bgr):
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


@app.route("/")
def index():
    samples = [f.name for f in sorted(SAMPLES_DIR.glob("*.jpg"))]
    return render_template("index.html", samples=samples)


@app.route("/predict", methods=["POST"])
def predict():
    filename = request.json.get("filename")
    if not filename:
        return jsonify(error="No image selected"), 400

    path = SAMPLES_DIR / filename
    img = cv2.imread(str(path))
    if img is None:
        return jsonify(error="Could not load image"), 400

    feat = extract_features(img).reshape(1, -1)
    feat = SCALER.transform(feat)

    proba = MODEL.predict_proba(feat)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = CLASSES[pred_idx]
    confidence = float(proba[pred_idx])

    action, tip = DISPOSAL[pred_class]

    return jsonify(
        prediction=pred_class,
        confidence=round(confidence, 3),
        action=action,
        tip=tip,
        probabilities={c: round(float(proba[i]), 3) for i, c in enumerate(CLASSES)},
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
