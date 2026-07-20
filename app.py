"""
app.py — MaizeScan Flask Application
=====================================
Author : Brice Gaetan Nono Youmbi | ULK Data Science 2025/2026
Supervisor: Prof. Jonas Niyitegeka

Routes:
    GET  /            → index.html  (upload form)
    POST /predict     → result.html (diagnosis)
    GET  /diseases    → diseases.html
    GET  /about       → about.html
    POST /api/predict → JSON API
    GET  /health      → JSON health check
"""

import os
import time
import logging
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from utils.leaf_image   import LeafImage
from utils.cnn_model    import CNNModel
from utils.diagnosis    import DiagnosisResult
from utils.disease_data import DISEASE_REGISTRY, CLASS_ORDER_KEYS

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("FlaskApp")

# ── Flask app ────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"]          = os.environ.get("SECRET_KEY", "maizescan-ulk-2026")
app.config["MAX_CONTENT_LENGTH"]  = 16 * 1024 * 1024   # 16 MB
app.config["UPLOAD_FOLDER"]       = os.path.join("static", "uploads")
app.config["ALLOWED_EXTENSIONS"]  = {"jpg", "jpeg", "png"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Model — load once at startup ─────────────────────────────────
MODEL_PATH = os.path.join("model", "vgg16_maize_best.h5")
IMG_SIZE   = (224, 224)
DEMO_MODE  = False

cnn = CNNModel(num_classes=4, img_size=IMG_SIZE)
if os.path.exists(MODEL_PATH):
    try:
        cnn.load(MODEL_PATH)
        log.info("Model ready.")
    except Exception as exc:
        log.warning("Model load failed: %s — DEMO MODE active", exc)
        DEMO_MODE = True
else:
    log.warning("Model file not found at '%s' — DEMO MODE active", MODEL_PATH)
    DEMO_MODE = True


# ── Helpers ───────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


def _demo_probs():
    """Fixed probabilities used in demo mode (Common Rust example)."""
    import numpy as np
    return np.array([0.05, 0.88, 0.04, 0.03], dtype=float)


# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", demo_mode=DEMO_MODE)


@app.route("/about")
def about():
    return render_template("about.html", demo_mode=DEMO_MODE)


@app.route("/diseases")
def diseases():
    return render_template(
        "diseases.html",
        diseases=DISEASE_REGISTRY,
        class_order=CLASS_ORDER_KEYS,
        demo_mode=DEMO_MODE,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Primary prediction endpoint.
    Sequence (Chapter 3 §3.6 Sequence Diagram):
      1. Receive multipart/form-data POST
      2. Validate file presence and extension
      3. Save to static/uploads/
      4. LeafImage.from_path() — preprocess
      5. CNNModel.predict()    — VGG16 inference
      6. DiagnosisResult.build_response() — format result
      7. Render result.html
    """
    # Step 1-2: validate
    if "leaf_image" not in request.files:
        return render_template(
            "index.html",
            error="No file received. Please choose a leaf image.",
            demo_mode=DEMO_MODE,
        )
    file = request.files["leaf_image"]
    if not file or file.filename == "":
        return render_template(
            "index.html",
            error="No file selected. Please choose a leaf image.",
            demo_mode=DEMO_MODE,
        )
    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="Invalid file type. Please upload a JPG or PNG image.",
            demo_mode=DEMO_MODE,
        )

    try:
        # Step 3: save
        safe_name = f"{int(time.time())}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(save_path)
        log.info("Saved upload: %s", save_path)

        # Step 4: preprocess (identical to Colab LeafImage)
        leaf = LeafImage.from_path(save_path, img_size=IMG_SIZE)

        # Step 5: inference
        probs = _demo_probs() if DEMO_MODE else cnn.predict(leaf.img_array)

        # Step 6: build full diagnosis response (with confidence score)
        result = DiagnosisResult(
            probs       = probs,
            class_order = CLASS_ORDER_KEYS,
            registry    = DISEASE_REGISTRY,
            leaf_image  = leaf,
        )
        response  = result.build_response()
        image_url = url_for("static", filename=f"uploads/{safe_name}")

        log.info(
            "Prediction: %s | Confidence: %s",
            response["prediction"],
            response["confidence_pct"],
        )

        # Step 7: render
        return render_template(
            "result.html",
            response  = response,
            image_url = image_url,
            demo_mode = DEMO_MODE,
        )

    except Exception as exc:
        log.error("Prediction error: %s", exc, exc_info=True)
        return render_template(
            "index.html",
            error=f"Processing error: {exc}. Please try a different image.",
            demo_mode=DEMO_MODE,
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API — same pipeline as /predict."""
    if "leaf_image" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["leaf_image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use JPG or PNG."}), 400
    try:
        safe_name = f"{int(time.time())}_{secure_filename(file.filename)}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(save_path)
        leaf  = LeafImage.from_path(save_path, img_size=IMG_SIZE)
        probs = _demo_probs() if DEMO_MODE else cnn.predict(leaf.img_array)
        result = DiagnosisResult(
            probs=probs, class_order=CLASS_ORDER_KEYS,
            registry=DISEASE_REGISTRY, leaf_image=leaf,
        )
        return jsonify(result.build_response())
    except Exception as exc:
        log.error("API error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "demo_mode":    DEMO_MODE,
        "model_loaded": not DEMO_MODE,
        "model_path":   MODEL_PATH,
        "class_order":  CLASS_ORDER_KEYS,
    })


# ── Error handlers ────────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    return render_template(
        "index.html", error="File too large. Max 16 MB.", demo_mode=DEMO_MODE
    ), 413


@app.errorhandler(404)
def not_found(e):
    return render_template(
        "index.html", error="Page not found.", demo_mode=DEMO_MODE
    ), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
