"""SortSmart Flask app backed by the selected deep vision checkpoint."""

import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for
from openai import OpenAI
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

from deep_vision import CLASSES, DeepImageClassifier


app = Flask(__name__)
MAX_UPLOAD_MB = int(os.environ.get("SORTSMART_MAX_UPLOAD_MB", "32"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

SAMPLES_DIR = Path("static/samples")

CHECKPOINT = Path(os.environ.get("SORTSMART_DEEP_CHECKPOINT", "output/deep/best_model.pt"))
DEVICE = os.environ.get("SORTSMART_DEEP_DEVICE", "auto")
CONFIDENCE_THRESHOLD = float(os.environ.get("SORTSMART_CONFIDENCE_THRESHOLD", "0.70"))
OPENAI_MODEL = os.environ.get("SORTSMART_OPENAI_MODEL", "gpt-5-mini")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

CLASSIFIER = None

DISPOSAL = {
    "cardboard": ("Recycle", "Blue bin: flatten clean cardboard first."),
    "glass": ("Recycle", "Blue bin: rinse containers and remove loose lids."),
    "metal": ("Recycle", "Blue bin: rinse cans and foil when accepted locally."),
    "paper": ("Recycle", "Blue bin: keep paper dry and free of food residue."),
    "plastic": ("Recycle", "Blue bin: rinse and check local plastic number rules."),
    "trash": ("Trash", "Black bin: general waste or contaminated material."),
}

CHAT_INSTRUCTIONS = """
You are SortSmart Assistant, a concise recycling guidance chatbot embedded in a class project web app.
The computer vision model is the source of truth for the predicted class and confidence scores.
Use the prediction context to answer the user's question, explain uncertainty when confidence is low,
and give practical disposal guidance. Keep answers under 120 words. Do not claim local rules with
certainty; tell users to check local guidance when rules can vary. Do not identify people or brands.
"""


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_classifier():
    global CLASSIFIER
    if CLASSIFIER is None and CHECKPOINT.exists():
        CLASSIFIER = DeepImageClassifier(CHECKPOINT, device=DEVICE)
    return CLASSIFIER


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    return jsonify(error=f"Image is too large. Use a file under {MAX_UPLOAD_MB} MB."), 413


def build_response(result, image_url=None, uploaded_filename=None):
    classifier = get_classifier()
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
        "model": classifier.checkpoint["model_key"],
        "threshold": CONFIDENCE_THRESHOLD,
    }


def build_chat_prompt(question, context, history):
    trimmed_history = history[-6:] if isinstance(history, list) else []
    return (
        "Prediction context:\n"
        f"{json.dumps(context, indent=2, sort_keys=True)}\n\n"
        "Recent chat messages:\n"
        f"{json.dumps(trimmed_history, indent=2, sort_keys=True)}\n\n"
        "User question:\n"
        f"{question}"
    )


def classify_path(image_path, image_url):
    classifier = get_classifier()
    return build_response(classifier.predict_path(image_path), image_url=image_url)


def classify_upload(upload):
    classifier = get_classifier()
    image = Image.open(upload.stream).convert("RGB")
    uploaded_filename = Path(upload.filename).name
    return build_response(classifier.predict_image(image), uploaded_filename=uploaded_filename)


@app.route("/")
def index():
    samples = [f.name for f in sorted(SAMPLES_DIR.glob("*.jpg"))]
    return render_template("index.html", samples=samples)


@app.route("/predict", methods=["POST"])
def predict():
    if get_classifier() is None:
        return jsonify(error=f"Model checkpoint is not installed at {CHECKPOINT}"), 503

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


@app.route("/chat", methods=["POST"])
def chat():
    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify(error="Set OPENAI_API_KEY to enable the recycling assistant."), 503

    payload = request.get_json(silent=True) or {}
    question = str(payload.get("message", "")).strip()
    context = payload.get("context") or {}
    history = payload.get("history") or []

    if not question:
        return jsonify(error="Ask a question first."), 400
    if not context.get("prediction"):
        return jsonify(error="Classify an image before using the assistant."), 400

    client = OpenAI()
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=CHAT_INSTRUCTIONS.strip(),
        input=build_chat_prompt(question[:1200], context, history),
        max_output_tokens=220,
    )
    return jsonify(reply=response.output_text, model=OPENAI_MODEL)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
