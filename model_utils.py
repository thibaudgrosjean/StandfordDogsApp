import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import pandas as pd
import numpy as np
from PIL import Image

tf.random.set_seed(999)

@st.cache(allow_output_mutation=True)
def prep_model(model_path, labels_path):

    model = tf.keras.models.load_model(model_path, compile=False)
    labels = pd.read_csv(labels_path, index_col=0)

    return model, labels

def preprocess_image(img_file_buffer):

    img = Image.open(img_file_buffer)
    img = img.resize((224, 224), Image.LANCZOS)
    img = img.convert('RGB')
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)

    return img_tensor

def get_prediction(model, labels, img_tensor):

    prediction = model.predict(img_tensor)
    label = labels.loc[np.argmax(prediction),'label']
    probability = str(round(prediction.max() * 100, 2))[:5]

    return label, probability