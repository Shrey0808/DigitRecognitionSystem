from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from prediction import preprocess_and_extract_digits
import os
import cv2

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)
            predictions, images = preprocess_and_extract_digits(img_path)
            
            processed_filenames = []
            for idx, img in enumerate(images):
                processed_filename = f"processed_{idx}.png"
                processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
                cv2.imwrite(processed_path, img)
                processed_filenames.append(processed_filename)
            
            return render_template('index.html', uploaded_filename=filename, processed_filenames=processed_filenames, predictions=predictions)
    return render_template('index.html')

app.jinja_env.globals.update(zip=zip)

if __name__ == '__main__':
    app.run(debug=True)
