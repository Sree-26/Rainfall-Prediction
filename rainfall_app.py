import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import warnings

warnings.filterwarnings('ignore')

# --- 1. Cache the Model and Scaler ---
# This function loads data, trains the model, and returns the trained model and scaler.
# @st.cache_resource ensures this function runs only once, making the app faster.
@st.cache_resource
def get_model_and_scaler():
    """
    Loads data, preprocesses it, and trains the SVC model.
    The output (model and scaler) is cached.
    """
    df = pd.read_csv('Rainfall.csv')

    # Clean column names and fill missing values
    df.rename(str.strip, axis='columns', inplace=True)
    for col in ['winddirection', 'windspeed']:
        if df[col].isnull().sum() > 0:
            val = df[col].mean()
            df[col] = df[col].fillna(val)
    df.replace({'yes': 1, 'no': 0}, inplace=True)

    # Feature Selection
    important_features = ['sunshine', 'humidity', 'cloud', 'pressure']
    features = df[important_features]
    target = df['rainfall']

    # Data splitting and resampling
    X_train, _, Y_train, _ = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=2
    )
    ros = RandomOverSampler(sampling_strategy='minority', random_state=22)
    X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)

    # Scaling and Training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    svc_model = SVC(kernel='rbf', probability=True)
    svc_model.fit(X_scaled, Y_resampled)

    return svc_model, scaler

# --- Load the cached model and scaler ---
model, scaler = get_model_and_scaler()


# --- 2. Set up the Streamlit User Interface ---
st.set_page_config(page_title="Rainfall Prediction", layout="wide")
st.title("🌧️ Rainfall Prediction App")
st.write("This app predicts the likelihood of rainfall using a machine learning model.")

# --- Sidebar for user inputs ---
st.sidebar.header("Weather Inputs")

sunshine = st.sidebar.number_input("Sunshine (in hours)", min_value=0.0, max_value=15.0, value=8.0, step=0.1)
humidity = st.sidebar.number_input("Humidity (in %)", min_value=0.0, max_value=100.0, value=73.0, step=0.5)
cloud = st.sidebar.number_input("Cloud Cover (in %)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
pressure = st.sidebar.number_input("Pressure (in hPa)", min_value=980.0, max_value=1050.0, value=1015.0, step=0.1)

# --- 3. Prediction Logic ---
if st.sidebar.button("Predict Rainfall"):
    # Create a DataFrame from inputs
    input_data = pd.DataFrame({
        'sunshine': [sunshine],
        'humidity': [humidity],
        'cloud': [cloud],
        'pressure': [pressure]
    })

    # Scale the input
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Display the result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("Yes, it is likely to rain. ☔")
    else:
        st.info("No, it is not likely to rain. ☀️")





