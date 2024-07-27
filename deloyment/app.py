from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from prediction import preprocess_and_predict

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error='No file part')

    img = request.files['image']
    
    if img.filename == '':
        return render_template('index.html', error='No selected file')

    if img and allowed_file(img.filename):
        filename = secure_filename(img.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(img_path)
        
        try:
            predictions, digit_images = preprocess_and_predict(img_path)
            results = [
                {'prediction': predictions[i], 'digit_image': digit_images[i]}
                for i in range(len(predictions))
            ]
            return render_template('index.html', results=results)
        except Exception as e:
            return render_template('index.html', error=f'Prediction failed: {str(e)}')

    return render_template('index.html', error='Invalid file format. Only PNG, JPG, JPEG, and GIF are allowed.')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
