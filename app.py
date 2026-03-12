from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

app = Flask(__name__)

# Replace 'best.pt' with the actual name of your weight file inside ai_engine
model = YOLO('model/best.pt')

@app.route('/')
def home():
    return "CassaDx AI API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    img_path = "temp_image.jpg"
    file.save(img_path)
    
    # Run YOLOv11 inference
    results = model(img_path)
    
    # Get the top prediction
    result = results[0]
    probs = result.probs
    top1_index = probs.top1
    label = result.names[top1_index]
    confidence = float(probs.top1conf)

    return jsonify({
        "status": label,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
