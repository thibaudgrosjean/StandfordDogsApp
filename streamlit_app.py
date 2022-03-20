import pandas as pd
import streamlit as st
import numpy as np
import time
import tensorflow

'Dogs race predictor'

uploaded_file = st.file_uploader('Choose picture file')

if uploaded_file is not None:

    with st.spinner('Predicting...'):

        st.image(uploaded_file)

        time.sleep(5)
        
        try: st.success('Caniche, probabilité: 56.08%')
        except: st.error('Error, please contact the assistance')
    # file =..
    # pred method
