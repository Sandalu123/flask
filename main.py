from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load the model
model = load_model('2_best_leaf_model.h5')

def adjust_brightness_and_saturation(image):
    image = tf.image.adjust_brightness(image, delta=0.2)
    image = tf.image.adjust_saturation(image, saturation_factor=1.2)
    return image
    
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'API is working!'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Load and preprocess image
    image = Image.open(request.files['image'].stream).resize((224, 224)) # Assuming the model takes 224x224 images
    image_arr = np.asarray(image) / 255.0  # Convert to numpy array and normalize
    image_arr = adjust_brightness_and_saturation(image_arr)
    image_arr = np.expand_dims(image_arr, axis=0)  # Expand dimensions for batch input
    
    # Predict
    predictions = model.predict(image_arr)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return jsonify({'class': str(predicted_class), 'confidence': str(confidence)})

if __name__ == '__main__':
    app.run(debug=True)
