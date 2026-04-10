import io
import json
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

try:
    from .label_info import DEFAULT_CLASS_ORDER, describe_class
except ImportError:
    from label_info import DEFAULT_CLASS_ORDER, describe_class

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MODEL_PATH = BASE_DIR / "skin_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
MAX_IMAGE_SIZE_BYTES = 8 * 1024 * 1024
IMAGE_SIZE = (224, 224)

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE_BYTES

_model = None
_model_error = None


def load_class_names():
    if CLASS_NAMES_PATH.exists():
        try:
            with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
                raw_data = json.load(file)
            if isinstance(raw_data, list):
                return raw_data
            if isinstance(raw_data, dict) and isinstance(raw_data.get("class_order"), list):
                return raw_data["class_order"]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return DEFAULT_CLASS_ORDER


class_names = load_class_names()


def get_model():
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error is not None:
        raise RuntimeError(_model_error)
    if not MODEL_PATH.exists():
        _model_error = (
            f"Model file not found at {MODEL_PATH}. Train the model first with "
            "`python backend/train.py`."
        )
        raise RuntimeError(_model_error)

    try:
        _model = load_model(MODEL_PATH)
        return _model
    except Exception as exc:  # pragma: no cover - depends on local TF runtime
        _model_error = f"Failed to load model: {exc}"
        raise RuntimeError(_model_error) from exc


def prepare_image(img, target_size=IMAGE_SIZE):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0


def format_top_predictions(predictions):
    top_indices = np.argsort(predictions)[::-1][:3]
    formatted = []
    for index in top_indices:
        code = class_names[index]
        details = describe_class(code)
        formatted.append(
            {
                "code": code,
                "name": details["name"],
                "description": details["description"],
                "confidence": round(float(predictions[index]), 6),
            }
        )
    return formatted


@app.get("/")
def index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return jsonify(
            {
                "message": "Skin disease classification API is running.",
                "frontend": "Frontend files were not found. Add frontend/index.html to serve the UI.",
            }
        )
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    model_ready = False
    error_message = None
    try:
        get_model()
        model_ready = True
    except RuntimeError as exc:
        error_message = str(exc)

    return jsonify(
        {
            "status": "ok",
            "modelReady": model_ready,
            "modelPath": str(MODEL_PATH),
            "classCount": len(class_names),
            "classes": [describe_class(code) for code in class_names],
            "error": error_message,
        }
    )


@app.get("/api/classes")
def get_classes():
    return jsonify({"classes": [describe_class(code) for code in class_names]})


@app.post("/api/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use the form field named 'file'."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        model = get_model()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    try:
        image_bytes = uploaded_file.read()
        if not image_bytes:
            return jsonify({"error": "Uploaded file is empty."}), 400

        img = Image.open(io.BytesIO(image_bytes))
        img_array = prepare_image(img)

        prediction_batch = model.predict(img_array, verbose=0)
        predictions = prediction_batch[0]
        predicted_index = int(np.argmax(predictions))
        predicted_code = class_names[predicted_index]
        predicted_details = describe_class(predicted_code)

        return jsonify(
            {
                "prediction": {
                    "code": predicted_code,
                    "name": predicted_details["name"],
                    "description": predicted_details["description"],
                    "confidence": round(float(predictions[predicted_index]), 6),
                },
                "topPredictions": format_top_predictions(predictions),
                "disclaimer": (
                    "This prototype is for educational use only and does not replace a medical diagnosis."
                ),
            }
        )
    except UnidentifiedImageError:
        return jsonify({"error": "Unsupported image format. Please upload a JPG or PNG image."}), 400
    except Exception as exc:  # pragma: no cover - runtime protection
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "Image is too large. Please upload a file smaller than 8 MB."}), 413


@app.get("/<path:path>")
def serve_static(path):
    file_path = FRONTEND_DIR / path
    if file_path.exists():
        return send_from_directory(app.static_folder, path)
    return jsonify({"error": "Resource not found."}), 404


if __name__ == "__main__":
    app.run(debug=True)
