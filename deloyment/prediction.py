import cv2
import numpy as np
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Load the MNIST model
model = load_model(r'C:\Users\KIIT\OneDrive - kiit.ac.in\Desktop\Desktop Items\comp\DigitRecognitionSystem\model\MNIST.h5')  

def preprocess_and_predict(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Grayscale
    
    if img is None:
        print(f"Error: Image file '{image_path}' not found or cannot be read.")
        return
    
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
            
            digit = cv2.resize(digit, (28, 28)) # Resize
            
            # Prepare image for model prediction
            digit_for_pred = digit.astype('float32')
            digit_for_pred /= 255
            digit_for_pred = np.expand_dims(digit_for_pred, axis=-1)
            digit_for_pred = np.expand_dims(digit_for_pred, axis=0)

            # Predict digit using the model
            prediction = model.predict(digit_for_pred)
            digit_class = np.argmax(prediction)
            predictions.append(digit_class)
            digit_images.append(digit)
    return predictions , digit_images


if __name__ =='__main__':
    image_path = 'static\img2.png' 
    predictions,digits = preprocess_and_predict(image_path)
    print("Predictions:", predictions)
