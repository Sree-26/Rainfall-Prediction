import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import warnings

warnings.filterwarnings('ignore')

# --- 1. Load and Prepare the Data ---
df = pd.read_csv('Rainfall.csv')

# Clean column names and fill missing values
df.rename(str.strip, axis='columns', inplace=True)
for col in ['winddirection', 'windspeed']:
    if df[col].isnull().sum() > 0:
        val = df[col].mean()
        df[col] = df[col].fillna(val)
df.replace({'yes': 1, 'no': 0}, inplace=True)

# --- 2. Feature Selection for a Simpler App ---
# We select the most important features to make the app easier to use.
important_features = ['sunshine', 'humidity', 'cloud', 'pressure']
features = df[important_features]
target = df['rainfall']

# --- 3. Train the Model on Selected Features ---
# Split the data
X_train, X_val, Y_train, Y_val = train_test_split(
    features,
    target,
    test_size=0.2,
    stratify=target,
    random_state=2
)

# Use RandomOverSampler to handle imbalance
ros = RandomOverSampler(sampling_strategy='minority', random_state=22)
X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)

# Normalize the selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Train the SVC model
svc_model = SVC(kernel='rbf', probability=True)
svc_model.fit(X_scaled, Y_resampled)


# --- 4. Create the Updated Prediction Function for Gradio ---
def predict_rainfall(sunshine, humidity, cloud, pressure):
    """
    Predicts rainfall based on the four most important features.
    """
    # Create a pandas DataFrame from the simplified inputs
    input_data = pd.DataFrame({
        'sunshine': [sunshine],
        'humidity': [humidity],
        'cloud': [cloud],
        'pressure': [pressure]
    })

    # Scale the input data using the same scaler from training
    scaled_input = scaler.transform(input_data)

    # Make a prediction
    prediction = svc_model.predict(scaled_input)

    # Return the result as a user-friendly string
    if prediction[0] == 1:
        return "Yes, it is likely to rain."
    else:
        return "No, it is not likely to rain."


# --- 5. Define the Simplified Gradio Interface ---
# Define the input components for the web interface with fewer fields
inputs = [
    gr.Number(label="Sunshine (in hours)"),
    gr.Number(label="Humidity (in %)"),
    gr.Number(label="Cloud Cover (in %)"),
    gr.Number(label="Pressure (in hPa)")
]

# Define the output component
output = gr.Textbox(label="Rainfall Prediction")

# --- 6. Launch the App ---
# Create and launch the Gradio interface
app = gr.Interface(
    fn=predict_rainfall,
    inputs=inputs,
    outputs=output,
    title="Rainfall Prediction App",
    description="Predict the likelihood of rainfall."
)

app.launch()



