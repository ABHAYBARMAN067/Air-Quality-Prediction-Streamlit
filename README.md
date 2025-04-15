# 🌫️ Air Quality Prediction App

A Streamlit-based web app that predicts the **Air Quality Index (AQI)** category using pollutant data like PM2.5, PM10, NO2, SO2, CO, and Temperature.

---

## 📌 Features

- 🧠 Predicts AQI using a trained **Random Forest Regressor**
- 🎯 Input pollutants and get real-time AQI category (e.g., Good, Poor, Hazardous)
- ⚙️ Built with **Streamlit**, **scikit-learn**, **joblib**
- 🔐 Clean and lightweight UI

---

## 📊 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python (Machine Learning)  
- **Libraries**: scikit-learn, pandas, numpy, joblib  

---

## 🏁 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/ABHAYBARMAN067/Air-Quality-Prediction-Streamlit.git
cd Air-Quality-Prediction-Streamlit


2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Streamlit app
bash
Copy
Edit
streamlit run streamlit_app.py
📦 Files in This Project

File	Description
streamlit_app.py	Streamlit app frontend/backend
air_quality_model.pkl	Trained RandomForest model
scaler.pkl	StandardScaler object
label_encoder.pkl	LabelEncoder for AQI categories
requirements.txt	Required Python libraries
updated_pollution_dataset.csv	Dataset used for training


