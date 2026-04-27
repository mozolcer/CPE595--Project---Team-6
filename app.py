"""SortSmart Flask app backed by the selected deep vision checkpoint."""

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

from deep_vision import CLASSES, DeepImageClassifier


app = Flask(__name__)
MAX_UPLOAD_MB = int(os.environ.get("SORTSMART_MAX_UPLOAD_MB", "32"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

SAMPLES_DIR = Path("static/samples")

CHECKPOINT = os.environ.get("SORTSMART_DEEP_CHECKPOINT", "output/deep/best_model.pt")
DEVICE = os.environ.get("SORTSMART_DEEP_DEVICE", "auto")
CONFIDENCE_THRESHOLD = float(os.environ.get("SORTSMART_CONFIDENCE_THRESHOLD", "0.70"))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

CLASSIFIER = DeepImageClassifier(CHECKPOINT, device=DEVICE)

DISPOSAL = {
    "cardboard": ("Recycle", "Blue bin: flatten clean cardboard first."),
    "glass": ("Recycle", "Blue bin: rinse containers and remove loose lids."),
    "metal": ("Recycle", "Blue bin: rinse cans and foil when accepted locally."),
    "paper": ("Recycle", "Blue bin: keep paper dry and free of food residue."),
    "plastic": ("Recycle", "Blue bin: rinse and check local plastic number rules."),
    "trash": ("Trash", "Black bin: general waste or contaminated material."),
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    return jsonify(error=f"Image is too large. Use a file under {MAX_UPLOAD_MB} MB."), 413


def build_response(result, image_url=None, uploaded_filename=None):
    prediction = result["prediction"]
    confidence = result["confidence"]
    probabilities = {
        cls: round(float(result["probabilities"][cls]), 4)
        for cls in CLASSES
    }
    top_predictions = [
        {"class": cls, "confidence": probabilities[cls]}
        for cls in sorted(CLASSES, key=lambda cls: probabilities[cls], reverse=True)[:3]
    ]
    action, tip = DISPOSAL[prediction]

    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "action": action,
        "tip": tip,
        "probabilities": probabilities,
        "top_predictions": top_predictions,
        "low_confidence": confidence < CONFIDENCE_THRESHOLD,
        "fallback": "Low confidence: retake with the item centered on a plain background, then compare the top predictions.",
        "image_url": image_url,
        "uploaded_filename": uploaded_filename,
        "model": CLASSIFIER.checkpoint["model_key"],
        "threshold": CONFIDENCE_THRESHOLD,
    }


def classify_path(image_path, image_url):
    return build_response(CLASSIFIER.predict_path(image_path), image_url=image_url)


def classify_upload(upload):
    image = Image.open(upload.stream).convert("RGB")
    uploaded_filename = Path(upload.filename).name
    return build_response(CLASSIFIER.predict_image(image), uploaded_filename=uploaded_filename)


@app.route("/")
def index():
    samples = [f.name for f in sorted(SAMPLES_DIR.glob("*.jpg"))]
    return render_template("index.html", samples=samples)


@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        filename = Path(str(payload.get("filename", ""))).name
        image_path = SAMPLES_DIR / filename
        if not filename or not image_path.exists():
            return jsonify(error="Sample image not found"), 400
        image_url = url_for("static", filename=f"samples/{filename}")
        return jsonify(classify_path(image_path, image_url))

    upload = request.files.get("image")
    if not upload or not upload.filename:
        return jsonify(error="Upload an image file first"), 400
    if not allowed_file(upload.filename):
        return jsonify(error="Supported formats: JPG, JPEG, PNG, WEBP"), 400

    return jsonify(classify_upload(upload))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
