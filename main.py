import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv('updated_pollution_dataset.csv')

print("Columns in the dataset:", df.columns)
print("Missing values in each column:")
print(df.isnull().sum())

# Data preprocessing
df = df.dropna()  # Remove rows with missing values
df = df.drop(columns=['O3'], errors='ignore')  # Drop 'O3' column if it exists

print("Columns after cleaning:", df.columns)

# Define target column and features
target_column = 'Air Quality'
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Temperature']

X = df[features]
y = df[target_column]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R² score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")

# Save the model, scaler, and label encoder
joblib.dump(model, 'air_quality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Shape of training set:", X_train.shape)
print("Shape of testing set:", X_test.shape)
