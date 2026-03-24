# 🌍 Air Quality Prediction Web Application
**Intelligent System Project (IS 2568)**

This web application predicts air quality factors using **Machine Learning (Ensemble)** and **Neural Networks** based on two different datasets. This project is developed as part of the Intelligent System course.

## 🌐 Live Demo
You can access the deployed application here: 
👉 **[https://intelligent-system---project-nz9tk7yjcrnydebnomnj5t.streamlit.app/]**

## 👥 Author
* **Name:** Mr. Thanakrit Muangrak
* **Student ID:** 6704062660183
* **University:** King Mongkut's University of Technology North Bangkok (KMUTNB)

## 📊 Datasets Info
1. **AirQualityUCI Dataset:** Predicts Relative Humidity (RH) using multi-sensor gas data from an Italian city.
2. **Bangkok Air Quality Dataset:** Predicts PM2.5 levels using monitoring data from the Pollution Control Department (PCD), Thailand.

> **Note on Data Quality:** Both datasets contained imprecision (missing values marked as -200 or NaN). These were handled during the preprocessing stage using Mean Imputation and Feature Scaling (StandardScaler) to ensure model accuracy.

## 🤖 Models Developed
* **Model 1 (Ensemble):** A **Voting Regressor** combining 3 distinct algorithms: Random Forest, Support Vector Regression (SVR), and Gradient Boosting.
* **Model 2 (Neural Network):** A **Sequential Deep Learning** model built with Keras/TensorFlow, optimized with ReLU activation and Adam optimizer.

## 📁 Project Structure
.
├── app.py                # Main Streamlit application code
├── requirements.txt      # List of Python dependencies
├── README.md             # Project documentation
└── models/               # Pre-trained models and scalers
    ├── IS_AirQuality_Dataset1/
    └── IS_BKKAirQuality_Dataset2/