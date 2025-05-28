from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import base64

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model
model = tf.keras.models.load_model('my_model.h5')

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Your main index page

# Route for About Us page
@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')  # About Us page

# Route for file upload and prediction
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (300, 300))  # Resize the image to match the model's input size
    img_resized_normalized = img_resized / 255.0  # Normalize
    img_resized_normalized = np.expand_dims(img_resized_normalized, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_resized_normalized)
    prediction_label = 'Pneumonia' if prediction >= 0.5 else 'Normal'

    if prediction_label == 'Pneumonia':
        # Convert to grayscale for processing
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred_img = cv2.GaussianBlur(gray_img, (15, 15), 0)
        
        # Apply a threshold to identify regions of interest (pneumonia-affected areas)
        _, thresholded_img = cv2.threshold(blurred_img, 50, 255, cv2.THRESH_BINARY)

        # Create a mask of the affected areas
        mask = cv2.bitwise_and(img, img, mask=thresholded_img)

        # Create a thick red marker to highlight the affected areas
        colored_mask = np.zeros_like(img)
        colored_mask[thresholded_img > 0] = [0, 0, 255]  # Red marker for affected area

        # Apply a thicker border (dilate the mask)
        kernel = np.ones((15, 15), np.uint8)  # Larger kernel for thicker borders
        dilated_mask = cv2.dilate(colored_mask, kernel, iterations=1)

        # Apply the dilated mask to the original image with a lower opacity to blend the highlight
        highlighted_img = cv2.addWeighted(img, 1, dilated_mask, 0.7, 0)
    else:
        # For normal X-rays, apply a light blue/gray overlay
        highlighted_img = img.copy()
        overlay_color = (230, 230, 250)  # Light blue or any color for normal images
        highlighted_img = cv2.addWeighted(highlighted_img, 0.9, np.full_like(highlighted_img, overlay_color), 0.2, 0)

    # Convert the image with highlights to a format that can be sent to the frontend
    _, img_encoded = cv2.imencode('.jpg', highlighted_img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return jsonify({
        'prediction': prediction_label,
        'highlighted_image': 'data:image/jpeg;base64,' + img_base64
    })

# Serve static files (for images, CSS, JS)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['TEMPLATES_AUTO_RELOAD'] = True

if __name__ == "__main__":
    app.run(debug=True)
