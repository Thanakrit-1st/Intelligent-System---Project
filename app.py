import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

st.sidebar.title("Main Menu")
page = st.sidebar.selectbox("Select Page", ["Datasets Info", "Machine Learning Model Details","Neural Network Model Details", "Test Machine Learning Model", "Test Neural Network Model"])

if page == "Datasets Info":
    st.header("TEst")
    st.write("")

elif page == "Machine Learning Model Details":
    st.header("")
    st.write("")
elif page == "Neural Network Model Details":
    st.header("")
    st.write("")

elif page == "Test Machine Learning Model":
    st.header("")
    st.write("")

elif page == "Test Neural Network Model":
    st.header("")
    st.write("")
