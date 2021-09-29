# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:01:19 2021

@author: mocta
"""

#To run this code, open the anaconda prompt, ensure we are in the right folder and launch the command:
# "streamlit run app.py"

import streamlit as st
import io
import tensorflow as tf
import numpy as np
from PIL import Image
import efficientnet.tfkeras as efn



st.title("Plant Disease Detection Leaf") # title of the web page
st.write("Just upload your plant Leaf image and get prediction if your plant is healthy or not") #text write

st.write("")

#Load model
model = tf.keras.models.load_model("model.h5") #import a model to predict. The model take as input 128*128 image with normalized value


#predicition Map

predicition_map = {0:"is healthy", 1:"has Multiple diseases", 2:"has rust",3:"has scab"}

#file uploader

uploaded_file = st.file_uploader("Choose your plant leaf image",type=['png', 'jpg']) #upload the image in binary format

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())) #read image in binary format with read and then transform it
    
    st.image(image,use_column_width=True) #show the image in the browser by adapting size to the window 
    
    resized_image = np.array(image.resize((512,512)))/255. #change the image to numpy array, resize and normalize it 
    
    #transform the image to a batch so that the model can be used
    image_batch = resized_image[np.newaxis,:,:,:]
    
    prediction_array = model.predict(image_batch) #prediction
    
    st.write(prediction_array) #print the prediction on the web 
    
    
    #convert a predicition array to a more beautiful thing
    
    prediction = np.argmax(prediction_array) #get the index of the max of prediction
    
    st.write(prediction)
    
    result_text = f"The plant leaf  {predicition_map[prediction]} with a probability of {int(prediction_array[0][prediction] *100)} %"
    
   # st.write(result_text)
    
    
    if prediction==0:
        st.success(result_text) #if pred = 0 so healthy, print result text in green
    else:
        st.error(result_text)  #print result text in red otherwise