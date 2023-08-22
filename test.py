# Importing Libraries
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import os

# importing model
model = tf.keras.models.load_model("models\carandbikemodel.h5")

# creating function
def classification(file):
    sample = cv2.imread(file)
    resize = tf.image.resize(sample, (256, 256))
    sample_pred = model.predict(np.expand_dims(resize / 255, 0))
    result = "Itzz a Car!!" if sample_pred > 0.5 else "Itz a Bike!!"
    return result

# Header
st.title("CAR AND BIKE CLASSIFIACTION")

st.divider()

# getting image
st.header("Please upload your Image")

uploaded_file = st.file_uploader(
    "Choose a file", type=["jpg", "jpeg", "png"], help="Upload an image"
)

# check for file
if uploaded_file is not None:    
        
    st.write(uploaded_file.name)
    st.image(uploaded_file)
    button = st.button("Predict")

    # if button is clicked
    if button:
        
        # creating path for the uploaded file
        image_path = './uploaded_files/'+uploaded_file.name
        # saving the file in the path
        with open(image_path,'wb') as f:
            f.write(uploaded_file.getbuffer())

        # progress bar    
        st.spinner('Wait for it...')
            
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            cv2.waitKey(10)
        
        # making classification
        result = classification(image_path)
        
        # displaying the result        
        st.success(result)
      
        # removing the file after prediction
        os.remove(image_path)
        
