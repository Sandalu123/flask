from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

def load_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        return str(e)

def adjust_brightness_and_saturation(image):
    image = tf.image.adjust_brightness(image, delta=0.2)
    image = tf.image.adjust_saturation(image, saturation_factor=1.2)
    return image

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'API is working!'}), 200

@app.route('/leaf_predict', methods=['POST'])
def predict():
    interpreter = load_model('2_best_leaf_model.tflite')
    if isinstance(interpreter, str):
        return jsonify({'error': interpreter}), 500
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Load and preprocess image
    image = Image.open(request.files['image'].stream).resize((250, 250))
    image_arr = np.asarray(image) / 255.0  # Convert to numpy array and normalize
    image_arr = adjust_brightness_and_saturation(image_arr)
    image_arr = np.expand_dims(image_arr, axis=0)  # Expand dimensions for batch input
    image_arr = image_arr.astype(np.float32)  # Cast to float32
    
    class_labels = ['disease1','disease2','disease3','healthy']

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], image_arr)
    
    # Invoke the interpreter to perform inference
    interpreter.invoke()

    # Retrieve the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    interpreter = None  # Clear the variable

    return jsonify({'class': str(class_labels[predicted_class]), 'confidence': str(confidence)})

@app.route('/sheet_predict', methods=['POST'])
def sheet_predict():
    interpreter = load_model('best_sheet.tflite')
    if isinstance(interpreter, str):
        return jsonify({'error': interpreter}), 500
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Load and preprocess image
    image = Image.open(request.files['image'].stream).resize((150, 150))
    image_arr = np.asarray(image) / 255.0  # Convert to numpy array and normalize
    image_arr = adjust_brightness_and_saturation(image_arr)
    image_arr = np.expand_dims(image_arr, axis=0)  # Expand dimensions for batch input
    image_arr = image_arr.astype(np.float32)  # Cast to float32
    
    sheet_class_labels = ['RSS2','RSS3','RSS4 or RSS5' ,'Sole Crepe', 'Thick Crepe']  # Adjust this list accordingly based on your model's classes

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], image_arr)
    
    # Invoke the interpreter to perform inference
    interpreter.invoke()

    # Retrieve the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    interpreter = None  # Clear the variable

    return jsonify({'class': str(sheet_class_labels[predicted_class]), 'confidence': str(confidence)})

if __name__ == '__main__':
    app.run(debug=True)