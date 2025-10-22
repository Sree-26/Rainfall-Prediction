# Rainfall Prediction using Machine Learning

## Overview

This project predicts rainfall based on key weather parameters using Machine Learning. It started as a basic model in Google Colab and was later deployed as an interactive **Streamlit web application** for real-time predictions.

## Features

* Predict rainfall probability using weather inputs
* Interactive user interface powered by Streamlit
* Feature selection to identify top predictors
* End-to-end workflow from data preprocessing to deployment

## Tech Stack

* **Python**
* **Pandas**, **NumPy**, **Scikit-learn**
* **Streamlit** for web app development
* **Google Colab** for initial model training

## Dataset

The dataset includes weather parameters such as:

* Sunshine (hours)
* Humidity (%)
* Cloud Cover (%)
* Pressure (hPa)

After feature selection, these four parameters were identified as the most influential in predicting rainfall.

## Model Development

1. Data cleaning and preprocessing
2. Feature selection and scaling
3. Model training using machine learning algorithms
4. Evaluation using performance metrics
5. Deployment on Streamlit for interactive predictions

## Installation

To run this project locally:

```bash
# Clone the repository
git clone https://github.com/Sree-26/Rainfall-Prediction
cd Rainfall-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run rainfall_app.py
```

## Project Structure

```
Rainfall-Prediction/
│
├── rainfall_app.py                # Streamlit web app
├── Rainfall_Prediction_ML.ipynb   # Model training notebook
├── requirements.txt               # Required packages
├── .devcontainer/                 # Dev container setup
└── README.md                      # Project documentation
```

## Deployment

The app is deployed on Streamlit Cloud for easy access:

* **Try it here:** [Rainfall Prediction App](https://rainfall-prediction-fspt2fdrxtadwq97ynaju6.streamlit.app/)
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
