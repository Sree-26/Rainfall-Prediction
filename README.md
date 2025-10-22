Rainfall Prediction Streamlit App 🌧️

This project takes a machine learning model trained to predict rainfall and deploys it as an interactive web application using Streamlit. The app allows users to input key weather parameters and receive a prediction on whether it is likely to rain.

Project Overview

The initial model was developed in a Jupyter Notebook using weather data from Rainfall.csv. This Streamlit application simplifies the user interface by focusing on the four most predictive features for rainfall in this dataset:

Sunshine: Duration of sunshine in hours.

Humidity: Relative humidity as a percentage.

Cloud Cover: Percentage of the sky covered by clouds.

Pressure: Atmospheric pressure in hPa.

The application uses a pre-trained Support Vector Classifier (SVC) model from scikit-learn.

Features

Interactive Interface: Allows users to input weather data via sliders.

Real-time Prediction: Provides immediate rainfall prediction ("Likely to Rain" or "Not Likely to Rain").

Feature Selection: Uses only the 4 most impactful features for a simpler user experience.

Data Preprocessing: Includes data scaling (StandardScaler) and handling of class imbalance (RandomOverSampler) as performed in the original analysis.

Caching: Uses Streamlit's caching (@st.cache_resource) to load and train the model only once, improving performance.

Setup and Usage

Follow these steps to run the application locally:

Clone the Repository (or download the files):

git clone <your-repository-link>
cd <your-repository-directory>


Create a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


Install Dependencies:
Make sure you have the Rainfall.csv file in the same directory as the script. Then, install the required Python packages:

pip install -r requirements.txt


Run the Streamlit App:

streamlit run rainfall_app_streamlit.py


Access the App:
Streamlit will provide a local URL (usually http://localhost:8501) in your terminal. Open this URL in your web browser to interact with the rainfall prediction app.

File Structure

rainfall_app_streamlit.py: The main Python script for the Streamlit application.

Rainfall.csv: The dataset used for training the model.

requirements.txt: Lists the necessary Python packages.

README.md: This file, providing information about the project.

Model Details

Algorithm: Support Vector Classifier (SVC) with an RBF kernel.

Libraries: scikit-learn, pandas, imbalanced-learn, streamlit.
