import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL1_PATH = os.path.join(BASE_DIR, "models", "IS_AirQuality_Dataset1", "ensemble_model.pkl")
NN1_PATH = os.path.join(BASE_DIR, "models", "IS_AirQuality_Dataset1", "nn_model.keras")
SCALER1_PATH = os.path.join(BASE_DIR, "models", "IS_AirQuality_Dataset1", "scaler.pkl")

MODEL2_PATH = os.path.join(BASE_DIR, "models", "IS_BKKAirQuality_Dataset2", "model1_ensemble_bkk.pkl")
NN2_PATH = os.path.join(BASE_DIR, "models", "IS_BKKAirQuality_Dataset2", "nn_bkk.keras")
SCALER2_PATH = os.path.join(BASE_DIR, "models", "IS_BKKAirQuality_Dataset2", "scaler_bkk.pkl")

@st.cache_resource
def load_models():
    try:
        with open(MODEL1_PATH, "rb") as f:
            ensemble_model = pickle.load(f)

        nn_model = load_model(NN1_PATH)

        with open(SCALER1_PATH, "rb") as f:
            scaler = pickle.load(f)

        with open(MODEL2_PATH, "rb") as f:
            ensemble_bkk = pickle.load(f)

        nn_bkk = load_model(NN2_PATH)

        with open(SCALER2_PATH, "rb") as f:
            scaler_bkk = pickle.load(f)

        return ensemble_model, nn_model, scaler, ensemble_bkk, nn_bkk, scaler_bkk

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None
    
ensemble_model, nn_model, scaler, ensemble_bkk, nn_bkk, scaler_bkk = load_models()

if ensemble_model is None:
    st.stop()

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
    st.write("The dataset consists of numerical (tabular) data and represents a regression problem because the goal is to predict a continuous target variable (RH for UCI and PM2.5 for Bangkok). Therefore, regression algorithms were selected.")

    st.subheader("2. Algorithms Used in the Ensemble")
    st.write("**1) Random Forest Regressor**")
    st.write("- Tree-based model")
    st.write("- Uses multiple decision trees")
    st.write("- Reduces overfitting through random sampling")
    st.write("- Suitable for nonlinear relationships")
    st.write("**Reference:**")
    st.write("- Breiman, L. (2001). Random Forests. Machine Learning.")

    st.write("**2) Support Vector Regressor (SVR)**")
    st.write("- Based on Support Vector Machine concepts")
    st.write("- Handles complex relationships using kernel functions")
    st.write("- Suitable for complex data patterns")
    st.write("**Reference:**")
    st.write("Cortes, C., & Vapnik, V. (1995). Support Vector Machines.")

    st.write("**3) K-Nearest Neighbors (KNN)**")
    st.write("- Predicts based on the average of nearest neighbors")
    st.write("- No assumption about data distribution")
    st.write("- Works well when data has spatial similarity")
    st.write("**Reference:**")
    st.write("- Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory.")

    st.write("**4) Voting Regressor (Ensemble Method)**")
    st.write("Voting Regressor combines multiple regression models by averaging their predictions:")
    st.latex(r"{FinalPrediction} = \frac{y_1 + y_2 + y_3}{3}")
    st.write("**Why Ensemble?**")
    st.write("- Reduces both bias and variance")
    st.write("- Improves generalization")
    st.write("- Increases prediction stability")
    st.write("**Reference:**")
    st.write("- Dietterich, T. G. (2000). Ensemble Methods in Machine Learning.")

    st.write("---")

    st.subheader("Data Preparation Process")
    st.write("**1) Handling Missing Values**")
    st.write("In the AirQualityUCI dataset:")
    st.write("- Missing values are represented as -200")
    st.write("- These values were converted to NaN")
    st.write("- Forward Fill (ffill) method was applied")
    st.write("**Reason:**")
    st.write("The dataset is time-series (hourly data), so previous values are strongly correlated with the next observation.")
    st.write("In the Bangkok Air Quality dataset:")
    st.write("- Missing values were inspected.")
    st.write("- Rows with missing PM2.5 values were removed.")
    st.write("- For other features, forward fill was applied when necessary.")
    st.write("**Reason:**")
    st.write("PM2.5 is the target variable. Removing rows without target values ensures correct supervised learning.")

    st.write("**2) Feature Scaling**")
    st.write("StandardScaler was applied:")
    st.latex(r"Z = \frac{X - \mu}{\sigma}")
    st.write("Where:")
    st.write("- μ = mean")
    st.write("- σ = standard deviation")
    st.write("**Reason:**")
    st.write("- SVR and KNN are sensitive to feature scale")
    st.write("- Neural Networks converge faster with normalized data")

    st.write("**3) Train-Test Split**")
    st.write("The dataset was split into:")
    st.write("- 80% Training")
    st.write("- 20% Testing")
    st.write("Purpose:")
    st.write("- Prevent data leakage")
    st.write("- Evaluate generalization performance")

    st.write("**4) Outlier Handling**")
    st.write("No explicit outlier removal was applied because:")
    st.write("- Ensemble methods are robust to noise")
    st.write("- Removing outliers may remove valuable extreme environmental conditions")

    st.write("---")

    st.header("Hyperparameters")

    st.subheader("AirQualityUCI Dataset")

    st.write("**Random Forest Hyperparameters:**")
    st.write("- n_estimators = 100")

    st.write("**SVR Hyperparameters:**")
    st.write("- kernel = 'rbf'")
    st.write("- C = 1.0")
    st.write("- epsilon = 0.1")

    st.write("**KNN Hyperparameters:**")
    st.write("- n_neighbors = 5")
    st.write("- weights = 'uniform'")

    st.subheader("Bangkok Air Quality Dataset")

    st.write("**Random Forest Hyperparameters:**")
    st.write("- n_estimators = 100")
    st.write("- random_state = 42")

    st.write("**SVR Hyperparameters:**")
    st.write("- kernel = 'rbf'")

    st.write("**KNN Hyperparameters:**")
    st.write("- n_neighbors = 5")

    st.write("---")

    st.header("Model Performance Results")
    st.write("**AirQualityUCI Dataset:**")
    st.write("- R²: 0.9144")
    st.write("**Bangkok Air Quality Dataset:**")
    st.write("- R²: 0.7583")


elif page == "Neural Network Model Details":
    st.header("Neural Network Model Details")

    st.write("---")

    st.write("The Neural Network model was designed based on the characteristics of the dataset, which consists of numerical (tabular) data and represents a regression problem (continuous value prediction).")

    st.write("---")

    st.subheader("**Architecture Design**")
    st.write("**AirQualityUCI Dataset**")
    st.write("Input Layer (11 nodes)")
    st.write("- Dense (64, ReLU)")
    st.write("- Dense (32, ReLU)")
    st.write("- Dense (16, ReLU)")
    st.write("- Output (1, Linear)")
    st.write("**Bangkok Dataset**")
    st.write("Input Layer (5 nodes)")
    st.write("- Dense (128, ReLU)")
    st.write("- Dense (64, ReLU)")
    st.write("- Dense (32, ReLU)")
    st.write("- Dense (16, ReLU)")
    st.write("- Output (1, Linear)")

    st.subheader("Activation Function : ReLU")
    st.latex(r"ReLU(x) = max(0, x)")
    st.write("Advantages:")
    st.write("- Prevents vanishing gradient problem")
    st.write("- Speeds up convergence")
    st.write("- Works well with tabular numerical data")

    st.write("---")

    st.subheader("1. Input Layer Design")
    st.write("The number of nodes in the input layer equals the number of dataset features to ensure the model receives complete information.")

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

    st.write("---")

    st.header("Hyperparameters")

    st.subheader("AirQualityUCI Dataset")

    st.write("Optimizer: Adam")
    st.write("Loss: MSE")
    st.write("Epochs: 50")
    st.write("Batch size: 32")
    st.write("Validation split: 0.2")

    st.subheader("Bangkok Air Quality Dataset")

    st.write("Architecture: 128-64-32-16-1 (ReLU)")
    st.write("Optimizer: Adam")
    st.write("Loss: MSE")
    st.write("Epochs: 100")
    st.write("Batch size: 32")
    st.write("Validation split: 0.2")
    st.write("EarlyStopping: patience=10, restore_best_weights=True")

    st.write("---")

    st.subheader("Model Performance Results")
    st.write("**AirQualityUCI Dataset:**")
    st.write("- Training Loss (MSE): 15.3983")
    st.write("- Training MAE: 2.9240")
    st.write("- Validation Loss (MSE): 18.7053")
    st.write("- Validation MAE: 3.2899")
    st.write("**Bangkok Air Quality Dataset:**")
    st.write("- Training Loss (MSE): 222.7170")
    st.write("- Training MAE: 10.0224")
    st.write("- Validation Loss (MSE): 326.9261")
    st.write("- Validation MAE: 12.3531")

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
            co = st.number_input("CO(GT)", min_value=0.0, value=2.6)
            nmhc = st.number_input("NMHC(GT)", min_value=0.0, value=150.0)
            c6h6 = st.number_input("C6H6(GT)", min_value=0.0, value=11.9)
            nox = st.number_input("NOx(GT)", min_value=0.0, value=166.0)
            no2 = st.number_input("NO2(GT)", min_value=0.0, value=113.0)
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

            scaled = scaler.transform(features)
            prediction = ensemble_model.predict(scaled)

            st.success(f"Predicted RH: {prediction[0]:.2f} %")

    # Bangkok
    else:

        st.subheader("Enter Bangkok Air Quality Parameters")

        col1, col2 = st.columns(2)

        with col1:
            pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=40.0)
            o3 = st.number_input("O3 (ppb)", min_value=0.0, value=20.0)

        with col2:
            no2 = st.number_input("NO2 (ppb)", min_value=0.0, value=15.0)
            so2 = st.number_input("SO2 (ppb)", min_value=0.0, value=5.0)
            co = st.number_input("CO (ppm)", min_value=0.0, value=0.8)

        if st.button("Predict PM2.5 in Bangkok (ML - Bangkok)"):

            features = np.array([[pm10, o3, no2, so2, co]])

            scaled = scaler_bkk.transform(features)
            prediction = ensemble_bkk.predict(scaled)

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

        st.subheader("Enter UCI Air Quality Parameters")

        col1, col2 = st.columns(2)

        with col1:
            co = st.number_input("CO(GT)", min_value=0.0, value=2.6, key="nn1")
            nmhc = st.number_input("NMHC(GT)", min_value=0.0, value=150.0, key="nn2")
            c6h6 = st.number_input("C6H6(GT)", min_value=0.0, value=11.9, key="nn3")
            nox = st.number_input("NOx(GT)", min_value=0.0, value=166.0, key="nn4")
            no2 = st.number_input("NO2(GT)", min_value=0.0, value=113.0, key="nn5")
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

            scaled = scaler.transform(features)
            prediction = nn_model.predict(scaled).flatten()

            st.success(f"Predicted RH: {prediction[0]:.2f} %")

    # Bangkok
    else:

        st.subheader("Enter Bangkok Air Quality Parameters")

        col1, col2 = st.columns(2)

        with col1:
            pm10 = st.number_input("PM10", min_value=0.0, value=40.0, key="bnn2")
            o3 = st.number_input("O3", min_value=0.0, value=20.0, key="bnn3")

        with col2:
            no2 = st.number_input("NO2", min_value=0.0, value=15.0, key="bnn4")
            so2 = st.number_input("SO2", min_value=0.0, value=5.0, key="bnn5")
            co = st.number_input("CO", min_value=0.0, value=0.8, key="bnn6")

        if st.button("Predict PM2.5 in Bangkok (NN - Bangkok)"):

            features = np.array([[pm10, o3, no2, so2, co]])

            scaled = scaler_bkk.transform(features)
            prediction = nn_bkk.predict(scaled).flatten()

            st.success(f"Predicted PM2.5: {prediction[0]:.2f} µg/m³")