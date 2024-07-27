import cv2
import numpy as np
from tensorflow.keras.models import load_model
import warnings
import base64
from io import BytesIO
from PIL import Image

warnings.filterwarnings('ignore')

# Load the MNIST model
model = load_model(r'..\model\MNIST.h5')  

def preprocess_and_predict(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Grayscale
    
    if img is None:
        raise FileNotFoundError(f"Image file '{image_path}' not found or cannot be read.")
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0) # Gaussian Blur
    
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) # adaptive thresholding
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # Find contours
    
    # Sort contours from left to right based on x coordinate of bounding box
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    digit_images = []
    predictions = []
        
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w > 10 and h > 10:
            digit = thresh[y:y+h, x:x+w] # Extract digit
            
            # Resize the digit to be smaller
            digit_resized = cv2.resize(digit, (20, 20))
            
            # Add a black border around the resized digit
            border_size = 4
            digit_with_border = cv2.copyMakeBorder(digit_resized, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            # Deepen white lines and darken the rest
            digit_with_border[digit_with_border > 0] = 255  # Deepen white lines
            digit_with_border[digit_with_border == 0] = 0   # Ensure black background
            
            # Resize to 28x28
            digit_with_border = cv2.resize(digit_with_border, (28, 28))
            
            # Prepare image for model prediction
            digit_for_pred = digit_with_border.astype('float32')
            digit_for_pred /= 255
            digit_for_pred = np.expand_dims(digit_for_pred, axis=-1)
            digit_for_pred = np.expand_dims(digit_for_pred, axis=0)

            # Predict digit using the model
            prediction = model.predict(digit_for_pred)
            digit_class = np.argmax(prediction)
            predictions.append(int(digit_class))  # Ensure native Python int type
            
            # Convert digit image to base64
            digit_pil = Image.fromarray(digit_with_border)
            buffered = BytesIO()
            digit_pil.save(buffered, format="PNG")
            digit_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            digit_images.append(digit_base64)
    
    return predictions, digit_images

if __name__ == '__main__':
    image_path = 'static/img2.png' 
    predictions, digits = preprocess_and_predict(image_path)
    print("Predictions:", predictions)
