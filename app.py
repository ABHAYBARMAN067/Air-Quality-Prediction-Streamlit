import os
from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler from 'models' directory
model = joblib.load(os.path.join('models', 'air_quality_model.pkl'))
scaler = joblib.load(os.path.join('models', 'scaler.pkl'))

# AQI categories
categories = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

# Route to display the form and predict air quality
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        pm25 = float(request.form['PM2.5'])
        pm10 = float(request.form['PM10'])
        no2 = float(request.form['NO2'])
        so2 = float(request.form['SO2'])
        co = float(request.form['CO'])

        # Create input feature array
        input_features = np.array([[pm25, pm10, no2, so2, co]])

        # Scale the features using the loaded scaler
        input_scaled = scaler.transform(input_features)

        # Predict the category
        prediction = model.predict(input_scaled)
        predicted_category = categories[prediction[0]]

        return render_template('index.html', prediction=predicted_category)

    return render_template('index.html', prediction=None)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
