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
    st.header("Machine Learning Model Details")

    st.write("---")

    st.subheader("1. Problem Type")
    st.write("The dataset consists of numerical (tabular) data and represents a regression problem because the goal is to predict continuous values such as Relative Humidity (RH). Therefore, regression algorithms were selected.")

    st.subheader("2. Reason for Choosing an Ensemble Model")
    st.write("An Ensemble Model combines predictions from multiple models to reduce error and improve accuracy.")
    st.write("Advantages of Ensemble:")
    st.write("- Reduces model variance")
    st.write("- Lowers the risk of overfitting")
    st.write("- Improves prediction stability")
    st.write("This project uses a Voting Regressor, which averages predictions from multiple models.")

    st.subheader("3. Algorithms Used in the Ensemble")
    st.write("**1) Random Forest Regressor**")
    st.write("- Tree-based model")
    st.write("- Uses multiple decision trees")
    st.write("- Reduces overfitting through random sampling")
    st.write("- Suitable for nonlinear relationships")

    st.write("**2) Support Vector Regressor (SVR)**")
    st.write("- Based on Support Vector Machine concepts")
    st.write("- Handles complex relationships using kernel functions")
    st.write("- Suitable for complex data patterns")

    st.write("**3) K-Nearest Neighbors (KNN)**")
    st.write("- Predicts based on the average of nearest neighbors")
    st.write("- No assumption about data distribution")
    st.write("- Works well when data has spatial similarity")

    st.subheader("4. Model Development Process")
    st.write("1. Handle Missing Values (-200 → NaN → Forward Fill)")
    st.write("2. Perform Feature Scaling using StandardScaler")
    st.write("3. Split data into Train/Test sets")
    st.write("4. Train each model individually")
    st.write("5. Combine models using Voting Regressor")
    st.write("6. Evaluate performance using MSE and MAE")

elif page == "Neural Network Model Details":
    st.header("Neural Network Model Details")

    st.write("---")

    st.write("The Neural Network model was designed based on the characteristics of the dataset, which consists of numerical (tabular) data and represents a regression problem (continuous value prediction).")

    st.subheader("1. Input Layer Design")
    st.write("The number of nodes in the input layer equals the number of dataset features (11 variables) to ensure the model receives complete information.")

    st.subheader("2. Hidden Layer Design")
    st.write("A multi-layer fully connected (Dense) architecture with ReLU activation functions was used.")
    st.write("Reasons for choosing ReLU:")
    st.write("- Reduces the vanishing gradient problem")
    st.write("- Enables learning of nonlinear relationships")
    st.write("- Suitable for numerical data")
    st.write("For the Bangkok dataset, the number of nodes was increased (128 → 64 → 32 → 16) to capture more complex patterns.")

    st.subheader("3. Output Layer")
    st.write("Uses 1 node because the model predicts a single continuous value. No activation function is applied to maintain linear output.")

    st.subheader("4. Optimizer")
    st.write("Adam optimizer was selected because:")
    st.write("- Automatically adjusts the learning rate")
    st.write("- Provides fast convergence")
    st.write("- Suitable for medium-sized datasets")

    st.subheader("5. Loss Function")
    st.write("Mean Squared Error (MSE) was used because it is standard for regression tasks and emphasizes larger errors.")

    st.subheader("6. Additional Technique")
    st.write("For the Bangkok dataset, EarlyStopping was applied to reduce overfitting by stopping training when validation loss no longer decreases.")

elif page == "Test Machine Learning Model":
    st.header("")
    st.write("")

elif page == "Test Neural Network Model":
    st.header("")
    st.write("")
