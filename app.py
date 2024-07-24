from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from skimage import exposure

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('face_expression_model.h5')

# Define the directory for uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the list of class labels
CLASS_LABELS = ['Angry', 'Ahegao', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Helper function for image preprocessing
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # Resize to match model input size
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian blur
    img = exposure.equalize_hist(img)  # Histogram equalization
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image and make prediction
        img = preprocess_image(filepath)
        predictions = model.predict(img)
        predicted_class = CLASS_LABELS[np.argmax(predictions)]

        return render_template('result.html', label=predicted_class, image_url=filepath)

if __name__ == '__main__':
    app.run(debug=True)
