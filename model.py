import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
try:
    data = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Error: 'test.csv' not found. Make sure the CSV file is in the same directory.")
    exit()

# Define features (X) and target (y)
features = ['location', 'age_group', 'month', 'spec_power']
target = 'quantity'

X = data[features]
y = data[target]

# Define categorical features for one-hot encoding
categorical_features = ['location', 'age_group']

# Create a preprocessor to handle categorical features
# OneHotEncoder converts categorical variables into a numerical format.
# 'remainder="passthrough"' ensures that numerical columns ('month', 'spec_power') are kept.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Create the model pipeline
# A pipeline chains together the preprocessor and the linear regression model.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate the model (optional)
score = model.score(X_test, y_test)
print(f"Model R^2 score: {score:.2f}")

# Save the trained model to a file
model_filename = 'test.pkl'
print(f"Saving model to {model_filename}...")
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print("Model saved successfully.")

# --- Example of how to load and use the model ---
print("\n--- Example Prediction ---")
# Load the model from the file
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Create a sample data point for prediction
# This simulates the input you would get from the web application
sample_input = pd.DataFrame({
    'location': ['Ruralville'],
    'age_group': ['Children'],
    'month': [6],  # Example: a program in June
    'spec_power': [-1.75] # Predicting for a specific eyeglass power
})

# Make a prediction
predicted_quantity = loaded_model.predict(sample_input)

print(f"Input: {sample_input.to_dict('records')[0]}")
# Predictions can be float, so we round to the nearest whole number for inventory
print(f"Predicted quantity needed: {predicted_quantity[0]:.0f}")
