import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('updated_pollution_dataset.csv')

# Print column names to check for any discrepancies
print("Columns in the dataset:", df.columns)

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Fill missing values or drop them (based on your preference)
df = df.dropna()  # or df.fillna(df.mean(), inplace=True)

# Drop the 'O3' column if it exists (this won't cause an error if the column is not found)
df = df.drop(columns=['O3'], errors='ignore')  # `errors='ignore'` will skip if 'O3' doesn't exist

# Print columns after cleaning
print("Columns after cleaning:", df.columns)

# Update the target column name to 'Air Quality' (AQI)
target_column = 'Air Quality'  # This is the correct target column name

# Select relevant features (now including 'Temperature')
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Temperature']
X = df[features]
y = df[target_column]  # Use 'Air Quality' as the target

# Encode the target variable (categorical to numeric)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train a model (RandomForest as an example)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using R² score, Mean Absolute Error (MAE), and Mean Squared Error (MSE)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print the evaluation metrics
print(f"R² score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")

# Save the trained model to a file
joblib.dump(model, 'air_quality_model.pkl')

# Save the scaler to be used in the Flask app later
joblib.dump(scaler, 'scaler.pkl')

# Print the shape of the dataset after processing
print("Shape of training set:", X_train.shape)
print("Shape of testing set:", X_test.shape)
