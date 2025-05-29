from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Thư mục lưu ảnh tạm thời
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model("model/driver_behavior_model_v4.h5")

# Danh sách nhãn hành vi
activity_map = ['Safe driving',
                'Texting - right',
                'Talking on the phone - right',
                'Texting - left',
                'Talking on the phone - left',
                'Operating the radio',
                'Drinking',
                'Reaching behind',
                'Hair and makeup',
                'Talking to passenger']

def predict_behavior(image_path):
    image = Image.open(image_path).resize((240, 240))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    pred_probabilities = model.predict(image)[0]
    pred = np.argmax(pred_probabilities)
    return f"{activity_map[pred]} (Confidence: {pred_probabilities[pred]:.2f})"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    
    result = predict_behavior(file_path)
    image_url = file_path.replace("\\", "/")  # For Windows path compatibility

    return jsonify({"result": result, "image_url": image_url})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
