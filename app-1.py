import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
MODEL_PATH = "mammals_model.h5"
model = load_model(MODEL_PATH)

# Get the class labels from the train data generator
class_labels = ['african_elephant', 'alpaca', 'american_bison', 'anteater', 'arctic_fox', 
                'armadillo', 'baboon', 'badger', 'blue_whale', 'brown_bear', 'camel']  # Adjust as needed

# Streamlit App
st.title("Mammals Image Classification")

st.write("Upload an image, and the model will predict the mammal species.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))  # Ensure the size matches your model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display the prediction
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
