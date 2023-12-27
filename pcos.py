import streamlit as st
from PIL import Image
import numpy as np
import joblib
from skimage.transform import resize
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Load the trained SVM model with GridSearchCV
grid_search_path = "C:/Users/Sazna/Downloads/svm_model.pkl"
grid_search_model = joblib.load(grid_search_path)

# Access the best estimator from the GridSearchCV
best_svm_model = grid_search_model.best_estimator_

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to a fixed size
    resized_image = resize(image, (150, 150, 3))
    
    # Flatten the resized image
    flat_data = resized_image.flatten()

    # Reshape to (1, -1) to handle varying image sizes
    flat_data = flat_data.reshape(1, -1)

    return flat_data

# Streamlit UI
st.title("PCOS Detection Web App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    
    # Display the size of the uploaded image
    st.write(f"Image Size: {image.size[0]}x{image.size[1]} pixels")

    st.write("Classifying...")

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Preprocess the image
    flat_data = preprocess_image(image_array)

    # Ensure that the number of features matches the model's expectations
    if flat_data.shape[1] == 67500:
        # Make a prediction
        prediction = best_svm_model.predict(flat_data)[0]
        
        # Display the result
        if prediction == 1:  # Adjust based on your model's label convention
            st.write("Prediction: PCOS Detected")
        else:
            st.write("Prediction: No PCOS Detected")
    else:
        st.warning("Invalid number of features in the input data. Check preprocessing.")
