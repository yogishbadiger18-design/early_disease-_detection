from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
MODEL_FILE = 'disease_model.h5'
IMG_SIZE = (150, 150)

# Class names (must match training order)
CLASS_NAMES = [
    "Coccidiosis",
    "Healthy",
    "Newcastle Disease",
    "Salmonella"
]

# Disease descriptions
DISEASE_INFO = {
    "Coccidiosis": {
        "description": "A parasitic disease that affects the intestinal tract of poultry.",
        "symptoms": "Diarrhea, weight loss, reduced egg production, lethargy",
        "severity": "Moderate to High",
        "color": "#ff6b6b"
    },
    "Healthy": {
        "description": "Bird appears to be in good health with no signs of disease.",
        "symptoms": "Active, normal appetite, bright eyes, clean feathers",
        "severity": "None",
        "color": "#51cf66"
    },
    "Newcastle Disease": {
        "description": "A highly contagious viral disease affecting respiratory and nervous systems.",
        "symptoms": "Respiratory distress, nervous signs, drop in egg production",
        "severity": "High",
        "color": "#ff8787"
    },
    "Salmonella": {
        "description": "A bacterial infection that can affect various organs in poultry.",
        "symptoms": "Depression, diarrhea, dehydration, reduced feed intake",
        "severity": "Moderate to High",
        "color": "#ffd43b"
    }
}

# Load model
try:
    model = load_model(MODEL_FILE)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_disease(img_path):
    """Predict the disease from image"""
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(prediction[0][predicted_class_idx])
        
        # Get all predictions with confidence scores
        predictions = []
        for i, class_name in enumerate(CLASS_NAMES):
            predictions.append({
                'class': class_name,
                'confidence': float(prediction[0][i])
            })
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': predictions,
            'disease_info': DISEASE_INFO.get(predicted_class, {})
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an image file.'})
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the model file.'})
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_disease(filepath)
        
        if 'error' in result:
            return jsonify(result)
        
        # Add file URL to result
        result['image_url'] = url_for('static', filename=f'uploads/{filename}')
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)