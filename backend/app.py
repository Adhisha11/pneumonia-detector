from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load your trained model
MODEL_PATH = "my_chest_xray_model.h5"  # make sure this file is in backend/
model = tf.keras.models.load_model(MODEL_PATH)

# Define your class names (must match training order)
CLASS_NAMES = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia", "COVID-19 Pneumonia"]

@app.route("/")
def home():
    return "✅ Pneumonia Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        # ⚡ Match training preprocessing
        img = img.resize((150, 150))  # Change to your training size
        img_array = np.array(img) / 255.0  # Normalize exactly as training
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]

        # ⚡ Correct class order from training
        CLASS_NAMES = ["COVID-19 Pneumonia","Normal",  "Bacterial Pneumonia","Viral Pneumonia" ]

        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions)) * 10

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)

