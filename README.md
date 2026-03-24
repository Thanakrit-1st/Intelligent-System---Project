# 🌍 Air Quality Prediction Web Application
**Intelligent System Project (IS 2568)**

This web application predicts air quality factors using **Machine Learning (Ensemble)** and **Neural Networks** based on two different datasets.

## 👥 Author
* **Name:** Mr. Thanakrit Muangrak
* **Student ID:** 6704062660183

## 📊 Datasets Info
1. **AirQualityUCI Dataset:** Predicts Relative Humidity (RH) using multi-sensor gas data from Italy.
2. **Bangkok Air Quality Dataset:** Predicts PM2.5 levels using monitoring data from the Pollution Control Department (PCD), Thailand.
*Note: Both datasets contained missing values (imprecision) and were preprocessed using mean imputation and feature scaling.*

## 🤖 Models Developed
* **Model 1 (Ensemble):** A Voting Regressor combining 3 models (Random Forest, SVR, and Gradient Boosting).
* **Model 2 (Neural Network):** A Sequential Deep Learning model designed specifically for each dataset's characteristics.

## 🛠️ How to Run Locally
1. Clone the repository:
   ```bash
   git clone <https://github.com/Thanakrit-1st/Intelligent-System---Project>