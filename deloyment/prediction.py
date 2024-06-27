import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the MNIST model
model = load_model(r'C:\Users\KIIT\OneDrive - kiit.ac.in\Desktop\Desktop Items\comp\DigitRecognitionSystem\model\MNIST.h5')  # Replace with your model file path

# Function to preprocess the image and extract digits
def preprocess_and_extract_digits(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Image file '{image_path}' not found or cannot be read.")
        return
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Perform adaptive thresholding to binarize the image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from left to right based on x coordinate of bounding box
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    digit_images = []
    predictions = []
    
    # Initialize figure for plotting
    
    # Iterate through contours and plot each digit
    for cnt in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Ensure bounding box isn't too small
        if w > 10 and h > 10:
            # Extract digit region
            digit = thresh[y:y+h, x:x+w]
            
            # Resize to a fixed size (e.g., 28x28) for model input
            digit = cv2.resize(digit, (28, 28))
            
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
    # Example usage:
    image_path = 'static\img1.jpg'  # Replace with your image path
    predictions = preprocess_and_extract_digits(image_path)
    print("Predictions:", predictions)
