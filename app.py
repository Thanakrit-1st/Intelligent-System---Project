import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

@st.cache_resource
def load_models():

    # Dataset 1
    with open("models/IS_AirQuality_Dataset1/ensemble_model.pkl", "rb") as f:
        ensemble1 = pickle.load(f)

    nn1 = load_model("models/IS_AirQuality_Dataset1/nn_model.keras", compile=False)

    with open("models/IS_AirQuality_Dataset1/scaler.pkl", "rb") as f:
        scaler1 = pickle.load(f)

    # Dataset 2
    with open("models/IS_BKKAirQuality_Dataset2/model1_ensemble_bkk.pkl", "rb") as f:
        ensemble2 = pickle.load(f)

    nn2 = load_model("models/IS_BKKAirQuality_Dataset2/nn_bkk.keras", compile=False)

    with open("models/IS_BKKAirQuality_Dataset2/scaler_bkk.pkl", "rb") as f:
        scaler2 = pickle.load(f)

    return ensemble1, nn1, scaler1, ensemble2, nn2, scaler2

ensemble1, nn1, scaler1, ensemble2, nn2, scaler2 = load_models()
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

    st.header("Test Machine Learning Model")
    st.write("---")

    dataset_choice = st.selectbox(
        "Select Dataset",
        ["AirQualityUCI Dataset", "Bangkok Air Quality Dataset"]
    )

    # AirQualityUCI
    if dataset_choice == "AirQualityUCI Dataset":

        st.subheader("Enter UCI Air Quality Parameters")

        col1, col2 = st.columns(2)

        with col1:
            co = st.number_input("CO(GT)", value=2.6)
            nmhc = st.number_input("NMHC(GT)", value=150.0)
            c6h6 = st.number_input("C6H6(GT)", value=11.9)
            nox = st.number_input("NOx(GT)", value=166.0)
            no2 = st.number_input("NO2(GT)", value=113.0)
            temp = st.number_input("Temperature (T)", value=13.6)

        with col2:
            s1 = st.number_input("PT08.S1(CO)", value=1360.0)
            s2 = st.number_input("PT08.S2(NMHC)", value=1046.0)
            s3 = st.number_input("PT08.S3(NOx)", value=1056.0)
            s4 = st.number_input("PT08.S4(NO2)", value=1692.0)
            s5 = st.number_input("PT08.S5(O3)", value=1268.0)

        if st.button("Predict Relative Humidity (RH) (ML - UCI)"):

            features = np.array([[co, s1, nmhc, c6h6, s2,
                                  nox, s3, no2, s4, s5, temp]])

            scaled = scaler1.transform(features)
            prediction = ensemble1.predict(scaled)

            st.success(f"Predicted RH: {prediction[0]:.2f} %")

    # Bangkok
    else:

        st.subheader("Enter Bangkok Air Quality Parameters")

        col1, col2 = st.columns(2)

        with col1:
            pm10 = st.number_input("PM10", value=40.0)
            o3 = st.number_input("O3", value=20.0)

        with col2:
            no2 = st.number_input("NO2", value=15.0)
            so2 = st.number_input("SO2", value=5.0)
            co = st.number_input("CO", value=0.8)

        if st.button("Predict PM2.5 in Bangkok (ML - Bangkok)"):

            features = np.array([[pm10, o3, no2, so2, co]])

            scaled = scaler2.transform(features)
            prediction = ensemble2.predict(scaled)

            st.success(f"Predicted PM2.5: {prediction[0]:.2f} µg/m³")

elif page == "Test Neural Network Model":
    st.header("Test Neural Network Model")
    st.write("---")

    dataset_choice = st.selectbox(
    "Select Dataset",
    ["AirQualityUCI Dataset", "Bangkok Air Quality Dataset"]
    )

    # AirQualityUCI
    if dataset_choice == "AirQualityUCI Dataset":

        st.subheader("Enter Bangkok Air Quality Parameters")

        col1, col2 = st.columns(2)

        with col1:
            co = st.number_input("CO(GT)", value=2.6, key="nn1")
            nmhc = st.number_input("NMHC(GT)", value=150.0, key="nn2")
            c6h6 = st.number_input("C6H6(GT)", value=11.9, key="nn3")
            nox = st.number_input("NOx(GT)", value=166.0, key="nn4")
            no2 = st.number_input("NO2(GT)", value=113.0, key="nn5")
            temp = st.number_input("Temperature (T)", value=13.6, key="nn6")

        with col2:
            s1 = st.number_input("PT08.S1(CO)", value=1360.0, key="nn7")
            s2 = st.number_input("PT08.S2(NMHC)", value=1046.0, key="nn8")
            s3 = st.number_input("PT08.S3(NOx)", value=1056.0, key="nn9")
            s4 = st.number_input("PT08.S4(NO2)", value=1692.0, key="nn10")
            s5 = st.number_input("PT08.S5(O3)", value=1268.0, key="nn11")

        if st.button("Predict Relative Humidity (RH) (NN - UCI)"):

            features = np.array([[co, s1, nmhc, c6h6, s2,
                                  nox, s3, no2, s4, s5, temp]])

            scaled = scaler1.transform(features)
            prediction = nn1.predict(scaled).flatten()

            st.success(f"Predicted RH: {prediction[0]:.2f} %")

    # Bangkok
    else:

        st.subheader("Enter Bangkok Air Quality Parameters")

        col1, col2 = st.columns(2)

        with col1:
            pm10 = st.number_input("PM10", value=40.0, key="bnn2")
            o3 = st.number_input("O3", value=20.0, key="bnn3")

        with col2:
            no2 = st.number_input("NO2", value=15.0, key="bnn4")
            so2 = st.number_input("SO2", value=5.0, key="bnn5")
            co = st.number_input("CO", value=0.8, key="bnn6")

        if st.button("Predict PM2.5 in Bangkok (NN - Bangkok)"):

            features = np.array([[pm10, o3, no2, so2, co]])

            scaled = scaler2.transform(features)
            prediction = nn2.predict(scaled).flatten()

            st.success(f"Predicted PM2.5: {prediction[0]:.2f} µg/m³")