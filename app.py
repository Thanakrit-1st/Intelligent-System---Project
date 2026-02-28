import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

st.sidebar.title("Main Menu")
page = st.sidebar.selectbox("Select Page", ["Datasets Info", "Machine Learning Model Details","Neural Network Model Details", "Test Machine Learning Model", "Test Neural Network Model"])

if page == "Datasets Info":
    st.title("Intelligent System Project")
    st.subheader("Air Quality Forecast")
    st.write("By: Mr. Thanakrit Muangrak 6704062660183")
    st.subheader("Datasets Used in This Project:")

    st.write("---")

    st.write("### 1) AirQualityUCI Dataset")
    st.write("This dataset contains air quality data collected from multi-sensor gas monitoring devices.")
    st.write("The sensors were installed in a city in Italy.")
    st.write("**Data recording period:** March 2004 – February 2005")
    st.write("**Location:** A highly polluted road area in an Italian city")
    st.write("**Data collected by:** Institute of Energy and the Environment (IIA-CNR), Italy")

    st.write("**Key variables in the dataset:**")
    st.write("- CO(GT): Hourly average concentration of Carbon Monoxide (mg/m³)")
    st.write("- NMHC(GT): Hourly average concentration of Non-Methane Hydrocarbons (μg/m³)")
    st.write("- C6H6(GT): Hourly average concentration of Benzene (μg/m³)")
    st.write("- NOx(GT): Hourly average concentration of Nitrogen Oxides (ppb)")
    st.write("- NO2(GT): Hourly average concentration of Nitrogen Dioxide (μg/m³)")
    st.write("- T: Temperature (°C)")
    st.write("- RH: Relative Humidity (%) ← Target Variable")

    st.markdown(
    '<a href="https://archive.ics.uci.edu/dataset/360/air+quality" target="_blank">**Source:** UCI Machine Learning Repository</a>',
    unsafe_allow_html=True
    )

    st.write("---")

    st.write("### 2) Bangkok Air Quality Dataset")
    st.write("Air quality dataset collected in Bangkok, Thailand.")

    st.write("**Data compiled by:** Prasert Kanawattanachai (GitHub: prasertcbs)")
    st.write("**Original data source:** Pollution Control Department (PCD), Thailand")

    st.write("**Key variables in the dataset:**")
    st.write("- PM2.5: Fine particulate matter ≤ 2.5 microns (μg/m³) ← Target Variable")
    st.write("- PM10: Particulate matter ≤ 10 microns (μg/m³)")
    st.write("- O3: Ozone")
    st.write("- NO2: Nitrogen Dioxide")
    st.write("- SO2: Sulfur Dioxide")
    st.write("- CO: Carbon Monoxide")

    st.markdown(
    '<a href="https://raw.githubusercontent.com/prasertcbs/basic-dataset/master/bangkok-air-quality.csv" target="_blank">**Source:** GitHub: prasertcbs</a>',
    unsafe_allow_html=True
    )

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
