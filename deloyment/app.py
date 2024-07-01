from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
from prediction import preprocess_and_predict
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Mock function for illustration, replace with actual implementation
def preprocess_and_predict(img_path):
    # Example: Loading image using PIL or OpenCV
    img_array = np.array([img_path])  # Replace with actual image processing logic
    predictions = np.array(['Prediction A', 'Prediction B'])  # Replace with actual predictions
    return img_array, predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST' and 'image' in request.files:
        img = request.files['image']
        if img.filename == '':
            return redirect(request.url)
        if img:
            filename = secure_filename(img.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(img_path)
            predictions,img_array = preprocess_and_predict(img_path)
            print(predictions)
            return jsonify(predictions.tolist())
    
    return 'Error in uploading image'

if __name__ == '__main__':
    app.run(debug=True)
