import os
import numpy as np
import cv2
import joblib
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Load MODELS
# -----------------------------

# Tabular model
TABULAR_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "tabular_rf_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "tabular_features.pkl")

rf_model = joblib.load(TABULAR_MODEL_PATH)
tabular_features = joblib.load(FEATURES_PATH)

# CNN model
CNN_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "cnn_model.h5")
cnn_model = load_model(CNN_MODEL_PATH)

IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "Conv_1"  # for MobileNetV2

# -----------------------------
# Utility Functions
# -----------------------------

def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# -----------------------------
# ROUTES
# -----------------------------

# Landing page


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/tabular")
def tabular_page():
    return render_template("tabular.html")

@app.route("/image")
def image_page():
    return render_template("image.html")


# -----------------------------
# TABULAR PREDICTION
# -----------------------------
@app.route("/predict_tabular", methods=["POST"])
def predict_tabular():

    user_input = []
    for feature in tabular_features:
        user_input.append(float(request.form[feature]))

    prediction = rf_model.predict([user_input])[0]
    probability = rf_model.predict_proba([user_input])[0][1]

    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    return render_template(
        "tabular.html",
        prediction=result,
        confidence=round(probability * 100, 2)
    )

# -----------------------------
# IMAGE PREDICTION + GRAD-CAM
# -----------------------------
@app.route("/predict_image", methods=["POST"])
def predict_image():

    if "image" not in request.files:
        return render_template("image.html")

    file = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    img_array = preprocess_image(image_path)

    pred = cnn_model.predict(img_array)[0][0]
    prediction = "Pneumonia" if pred > 0.5 else "Normal"
    confidence = round(float(pred if pred > 0.5 else 1 - pred) * 100, 2)

    heatmap = make_gradcam_heatmap(img_array, cnn_model, LAST_CONV_LAYER)

    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    gradcam_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    gradcam_path = os.path.join(OUTPUT_FOLDER, "gradcam.png")
    cv2.imwrite(gradcam_path, gradcam_img)

    return render_template(
        "image.html",
        prediction=prediction,
        confidence=confidence,
        gradcam_path="outputs/gradcam.png"
    )

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)