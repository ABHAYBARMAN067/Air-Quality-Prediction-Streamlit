from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and label encoder only once at startup
try:
    model = joblib.load('air_quality_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')  # Load label encoder here
    print("Model, Scaler, and Label Encoder loaded successfully.")
except Exception as e:
    print(f"Error loading model, scaler, or label encoder: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how-to-use')
def how_to_use():
    return render_template('how_to_use.html')

@app.route('/parameters')
def parameters():
    return render_template('parameters.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pm25 = float(request.form['PM2.5'])
        pm10 = float(request.form['PM10'])
        no2 = float(request.form['NO2'])
        so2 = float(request.form['SO2'])
        co = float(request.form['CO'])
        temperature = float(request.form['Temperature'])

        # Validate input
        if any(x < 0 for x in [pm25, pm10, no2, so2, co, temperature]):
            raise ValueError("Pollutant values can't be negative.")

        features = np.array([[pm25, pm10, no2, so2, co, temperature]])
        features_scaled = scaler.transform(features)

        predicted_aqi = model.predict(features_scaled)

        # Decode prediction using pre-loaded label encoder
        predicted_category = label_encoder.inverse_transform([int(round(predicted_aqi[0]))])[0]

        return render_template('index.html', prediction_text=f"The predicted Air Quality is: {predicted_category}")
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return render_template('index.html', prediction_text=f"Invalid input value: {ve}")
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
