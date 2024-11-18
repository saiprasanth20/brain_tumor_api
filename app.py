from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('brain_tumor_detector.h5')

# Upload folder for storing images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and prediction."""
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction="No file uploaded!")
        
        file = request.files['image']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image
            img = cv2.imread(file_path)
            img = cv2.resize(img, (224, 224))  # Resize image to match model input size
            img = img / 255.0  # Normalize image (if the model expects this)
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Debug: Check the shape of the image
            print(f"Image shape after preprocessing: {img.shape}")

            # Make prediction
            prediction = model.predict(img)
            
            # Debug: Check the model's output
            print(f"Prediction raw output: {prediction}")

            # Check the prediction value and determine result
            result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
            
            # Return prediction result and image path to render template
            return render_template('index.html', prediction=result, image_path=file_path)
        else:
            return render_template('index.html', prediction="No file selected!")
    else:
        return render_template('index.html', prediction="Invalid request!")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
