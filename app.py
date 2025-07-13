from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS to allow requests from your web application's domain
CORS(app)

# Load the pre-trained model
try:
    with open('test.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print("Error: 'test.pkl' not found. Please run train_model.py first.")
    model = None

@app.route('/')
def home():
    return "Inventory Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict inventory needs.
    Expects a JSON payload with 'location', 'age_group', and 'month'.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    # Get data from the POST request
    data = request.get_json()
    print(f"Received data for prediction: {data}")

    if not all(key in data for key in ['location', 'age_group', 'month']):
        return jsonify({'error': 'Missing required fields: location, age_group, month'}), 400

    location = data['location']
    age_group = data['age_group']
    month = int(data['month'])

    # Define a standard range of spectacle powers to predict for.
    # This is based on common prescriptions.
    # Myopia (nearsightedness) is common in children, Presbyopia (farsightedness) in seniors.
    if age_group == 'Children':
        spec_powers = [-0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0]
    elif age_group == 'Adults':
        spec_powers = [-1.0, -0.75, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    else: # Seniors
        spec_powers = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

    # Create a DataFrame for prediction
    prediction_input = pd.DataFrame({
        'location': [location] * len(spec_powers),
        'age_group': [age_group] * len(spec_powers),
        'month': [month] * len(spec_powers),
        'spec_power': spec_powers
    })

    # Make predictions
    try:
        predicted_quantities = model.predict(prediction_input)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


    # Format the results
    results = []
    for i, power in enumerate(spec_powers):
        # Ensure quantity is not negative and round to the nearest integer
        quantity = max(0, round(predicted_quantities[i]))
        results.append({
            'spec_power': f"{power:+.2f}", # Format power with sign (e.g., +1.25)
            'predicted_quantity': int(quantity)
        })
    
    print(f"Prediction results: {results}")
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
