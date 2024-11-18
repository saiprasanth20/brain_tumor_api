from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model('brain_tumor_detector.h5')

@app.route('/')
def home():
    return 'Welcome to the Brain Tumor Prediction API!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the file from the request
        file = request.files['image']
        
        # Convert the file to an OpenCV image format
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Resize image to match model's expected input
        img = cv2.resize(img, (224, 224))  # Resize to 224x224
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = img.astype('float32') / 255  # Normalize pixel values to [0, 1]

        # Add batch dimension to image: (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)

        # Check the image shape after processing
        print(f"Image shape after processing: {img.shape}")

        # Perform prediction
        prediction = model.predict(img)

        # Return the prediction result (you can customize this as per your output)
        return jsonify({'prediction': str(prediction[0][0])})
    
    except Exception as e:
        # Handle any exceptions
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == "__main__":
    # Define the port (you can adjust if necessary)
    app.run(debug=True, host='0.0.0.0', port=10000)
