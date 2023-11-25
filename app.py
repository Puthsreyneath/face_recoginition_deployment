import streamlit as st
from joblib import load
import cv2
import numpy as np
from PIL import Image
import re

# Load the model, PCA and Scaler
loaded_model = load('lda_clf.pkl')
pca = load('pca.pkl')
scaler = load('scaler.pkl')

# create rezise and gray_scale function for preprocessing the input image
# Assuming you have a global variable for the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Assuming you have a global variable for the target size of the detected face
target_size = (100, 100)

def detect_single_face(image):
    # Detect faces
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        # Crop and resize the face to a fixed size (e.g., 100x100)
        detected_face = cv2.resize(image[y:y+h, x:x+w], target_size)
        return detected_face

    # Return None if no face or more than one face is detected
    return None

def gray_scale(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return gray_image




# Function to preprocess and extract features from an input image
def preprocess_and_extract_features(image):
    # Convert the image to grayscale
    gray_image = gray_scale(image)

    # Resize the image to the target size
    detected_image = detect_single_face(gray_image)

    # check if no face or more than one face is detected
    if detected_image is None:
            return None
    
    # Flatten and scale the image
    flattened_image = detected_image.reshape(1, -1)
    scaled_image = scaler.transform(flattened_image)

    # Apply PCA transformation
    transformed_image = pca.transform(scaled_image)

    return transformed_image

# Function to extract name from image name
def extract_name(image_name):
    # Assuming the image name format is "FirstName_LastName_0001.jpg"
    match = re.match(r"([a-zA-Z]+_[a-zA-Z]+)_\d+.jpg", image_name)
    if match:
        return match.group(1)
    else:
        return None


# Streamlit app
def main():
    # Name the title for display in streamlit interface
    st.title("Face Recognition Using LDA & PCA")
    # specified group name
    st.markdown(
                """
                <style>
                    .footer {
                        text-align: right;
                        font-size: 12px;
                        color: gray;
                    }
                </style>
                <div class="footer" style='font-size:15px;color:brown;font-weight:bold;'>Created by Group 3</div>
                """,
                unsafe_allow_html=True
            )

    # Add HTML and CSS to customize the file uploader color
    st.markdown(
        """
        <style>
            .st-emotion-cache-1gulkj5 { 
                background-color: #E8F89F !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # File uploader for user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the custom file uploader display using Markdown
        st.markdown(f"### Uploaded Image: {uploaded_file.name}")

        # Display the uploaded image
        st.image(uploaded_file, caption=f"Original Image: {uploaded_file.name}", use_column_width=True)

        # Extract name from the image name
        image_name = uploaded_file.name
        extracted_name = extract_name(image_name)
        image = Image.open(uploaded_file)

        # Preprocess and extract features from the uploaded image
        features = preprocess_and_extract_features(image)

        # check if no face or more than one face is detected
        if features is None:
            st.markdown(f"### <span style='color:red;'>No Face or More than One Face is detected!</span>", unsafe_allow_html=True)
            return 
        
        # Make predictions using the loaded SVM model
        prediction = loaded_model.predict(features)[0]

        # Display the prediction result
        # if the prediction match the true label, the text will display in green color
        if prediction == extracted_name:
            label_color = "green"
        else:
            label_color = "red"
        st.markdown(f"### Prediction: <span style='color:{label_color};'>{prediction}</span>", unsafe_allow_html=True)


# run main to start the app        
if __name__ == '__main__':
    main()
