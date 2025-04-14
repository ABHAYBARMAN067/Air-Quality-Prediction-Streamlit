# Air Quality Prediction ML Project
# Problem: Predict Air Quality Category based on pollutant values (PM2.5, PM10, NO2, SO2, CO, O3)
# Type of Problem: Classification  
# Input: PM2.5, PM10, NO2, SO2, CO, O3
# Output: AQI Category
# ➡️ This is a classification problem

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('updated_pollution_dataset.csv')

# Check first 5 rows
print(df.head())

# Check dataset shape
print("Dataset shape:", df.shape)

# Check column names
print("Columns:", df.columns)

# Check data types
print("Data types:\n", df.dtypes)

# Check for null values
print("Missing values per column:\n", df.isnull().sum())

# Basic stats
print(df.describe())

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['number'])

# Correlation matrix of numeric columns
print("Correlation Matrix:\n", numeric_df.corr())

# Heatmap for correlation
sns.heatmap(numeric_df.corr(), annot=True)
plt.show()

# FEATURE ENGINEERING
# Select input features and target variable
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']]  # 'O3' nahi hai, 'Air Quality' ko target banayenge
y = df['Air Quality']  # 'Air Quality' ko target ke liye use karenge

# Check unique categories in target
print("Unique AQI Categories:", y.unique())

# LABEL ENCODING
# Convert categorical labels to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Check encoded labels
print("Encoded AQI Categories:", y_encoded)
print("Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# FEATURE SCALING
# Initialize StandardScaler
scaler = StandardScaler()

# Scale the features
X_scaled = scaler.fit_transform(X)

# Check the scaled features
print("Scaled Features:\n", X_scaled[:5])  # Show first 5 rows of scaled data

# TRAIN-TEST SPLIT
# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Check the shape of the split data
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)    

# MODEL SELECTION & TRAINING
# We will use Random Forest Classifier for this task, as it’s a robust and commonly used model for classification problems.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model for future use
joblib.dump(model, 'air_quality_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
