import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Paths
model_file = 'disease_model.h5'
img_size = (150, 150)

# List your classes (Must match training order!)
class_names = [
    "cocci",
    "healthy",
    "ncd",
    "salmo"
    # add more if needed
]

# Load Model
model = load_model(model_file)

def predict_image(img_path):
    """Predict the class of a given image"""
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    print(f"Predicted disease: {predicted_class}")

# Usage Example
predict_image(r"D:\POULTRY_DISEASDE\test\cocci\cocci.0.jpg")
