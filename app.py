from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
try:
    model = joblib.load('air_quality_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

@app.route('/')
def home():
    return render_template('index.html')  # Home page

@app.route('/about')
def about():
    return render_template('about.html')  # About page

@app.route('/how-to-use')
def how_to_use():
    return render_template('how_to_use.html')  # How to Use page

@app.route('/parameters')
def parameters():
    return render_template('parameters.html')  # Parameters page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the user (PM2.5, PM10, NO2, SO2, CO, Temperature)
        pm25 = float(request.form['PM2.5'])
        pm10 = float(request.form['PM10'])
        no2 = float(request.form['NO2'])
        so2 = float(request.form['SO2'])
        co = float(request.form['CO'])
        temperature = float(request.form['Temperature'])

        # Prepare the features for prediction
        features = np.array([[pm25, pm10, no2, so2, co, temperature]])

        # Debug print statement to check the features being passed
        print(f"Features for prediction: {features}")

        # Scale the features using the saved scaler
        features_scaled = scaler.transform(features)

        # Predict the Air Quality Index (AQI)
        predicted_aqi = model.predict(features_scaled)
        
        # Debug print statement to check prediction
        print(f"Predicted AQI (numeric value): {predicted_aqi}")

        # Decode the predicted numeric label to its corresponding category
        aqi_category = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
        
        # Handle cases where prediction might be slightly out of range (e.g., rounding errors)
        predicted_category = aqi_category[int(np.round(predicted_aqi[0]))]

        return render_template('index.html', prediction_text=f"The predicted Air Quality is: {predicted_category}")
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return render_template('index.html', prediction_text=f"Invalid input value: {ve}")
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
