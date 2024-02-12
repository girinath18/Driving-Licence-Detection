import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model for driving license detection
model = load_model(r"C:\Users\Dell 5470\OneDrive\Desktop\Streamlit Web Apps Store\Driving Licence Using Streamlit\models\Dynamite.h5")  

def main():
    st.title("Driving Licence Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Display the original image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for prediction
        resized_image = cv2.resize(image, (128, 128))
        scaled_image = resized_image / 255.0
        reshaped_image = np.reshape(scaled_image, [1, 128, 128, 3])

        # Make prediction
        prediction = model.predict(reshaped_image)
        pred_label = np.argmax(prediction)

        # Display prediction result
        if pred_label == 1:
            st.success('Driving license detected!')
        else:
            st.warning('No driving license detected.')

if __name__ == "__main__":
    main()
