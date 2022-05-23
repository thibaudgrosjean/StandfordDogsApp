from codecs import xmlcharrefreplace_errors
from multiprocessing.sharedctypes import Value
import streamlit as st
from model_utils import prep_model, preprocess_image, get_prediction
import gdown
import os

MODEL_PATH = './model.h5'
LABELS_PATH = './labels.csv'

def main():
    st.title('Whatadog predictor')
    img_file_buffer = st.file_uploader('Choose picture file')
    preview = st.empty()
    success = st.empty()

    # Load or download model
    if os.path.exists(MODEL_PATH):
        pass
    else:
        with st.spinner('Downloading Model file...'):
            url = 'https://drive.google.com/uc?id=1DWptw2lppif-Po85CHiHqZZi3_nbsomB'
            output = 'model.h5'
            gdown.download(url, output, quiet=False)
            st.success('Model downloaded')

    model, labels = prep_model(MODEL_PATH, LABELS_PATH)

    # Manage Drag & Drop
    if img_file_buffer is not None:
        with st.spinner('Predicting...'):
            try: 
                img_tensor = preprocess_image(img_file_buffer)
                label, probability = get_prediction(model, labels, img_tensor)
                preview.image(img_file_buffer, width=300)
                success.success(f'• Prediction: {label} • Probability: {probability}%')
            except: st.error('Error, please contact the assistance')

if __name__ == "__main__":
    main()
