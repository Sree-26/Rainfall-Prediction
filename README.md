# 🌧️ Rainfall Prediction using Machine Learning

**A machine learning solution to predict rainfall probability based on meteorological data, deployed as an interactive web app.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📖 Overview
This project leverages Machine Learning to predict whether it will rain tomorrow based on specific weather conditions today. It began as an exploratory data analysis and model training experiment in **Google Colab** and evolved into a production-ready web application using **Streamlit**.

The goal is to provide a simple, user-friendly interface where users can input weather parameters and receive an instant prediction.

## ✨ Features
* **Real-time Prediction:** Instant classification (Rain / No Rain) based on user inputs.
* **Interactive UI:** Built with Streamlit for a seamless user experience.
* **Optimized Features:** Uses a selected subset of high-impact weather parameters.
* **End-to-End Pipeline:** Covers data cleaning, preprocessing, imbalance handling, training, and deployment.

## 🧠 Model & Methodology

### 1. The Dataset & Feature Selection
The model was trained on historical weather data. Through correlation analysis and feature selection techniques, the following **4 key parameters** were identified as the strongest predictors of rainfall:
* ☀️ **Sunshine:** Daily sunshine duration (hours).
* 💧 **Humidity:** Relative humidity percentage (%).
* ☁️ **Cloud Cover:** Percentage of sky covered by clouds (%).
* 🎈 **Pressure:** Atmospheric pressure (hPa).

### 2. Algorithm Details
* **Model:** Support Vector Classifier (SVC).
* **Kernel:** Radial Basis Function (RBF) – chosen for its ability to handle non-linear relationships in weather data.
* **Data Handling:** Utilized `imbalanced-learn` to handle class imbalance (ensuring the model doesn't just predict "No Rain" due to majority class bias).

## 🛠️ Tech Stack
* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-learn, Imbalanced-learn
* **Data Manipulation:** Pandas, NumPy
* **Environment:** Google Colab (Training), VS Code (Development)

## 📂 Project Structure
```bash
Rainfall-Prediction/
│
├── .devcontainer/                 # Configuration for dev environments
├── rainfall_app.py                # Main Streamlit application file
├── Rainfall_Prediction_ML.ipynb   # Jupyter Notebook for EDA & Training
├── requirements.txt               # List of dependencies
└── README.md                      # Project documentation
```

## Deployment

The app is deployed on Streamlit Cloud for easy access:

* **Try it here:** [Rainfall Prediction App](https://rainfall-prediction-sree.streamlit.app/)
* **Source Code:** [GitHub Repository](https://github.com/Sree-26/Rainfall-Prediction/tree/main)

## Future Enhancements

* Add visualization for rainfall trends
* Integrate weather API for live data input
* Experiment with deep learning models for improved accuracy

## License

This project is licensed under the **MIT License**.

---

**Author:** Sree | *Machine Learning Enthusiast*

#MachineLearning #Streamlit #Python #AI #DataScience #AIML


Model Details

Algorithm: Support Vector Classifier (SVC) with an RBF kernel.

Libraries: scikit-learn, pandas, imbalanced-learn, streamlit.
