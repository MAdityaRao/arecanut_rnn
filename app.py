
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from flask_cors import CORS
import os
import joblib

# --- Step 1: Initialize Flask App ---
app = Flask(__name__)
# IMPORTANT: When running locally, ensure you use the exact port 5001 in the front-end fetch calls.
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Step 2: Define Model and Data Configuration ---
CONFIG = {
    'adike': {
        'model_path': 'model_adike.h5',
        'scaler_path': 'scaler_adike.gz',
        'csv_path': 'arecanut.csv', 
        'price_column': 'Max Price (Rs./Quintal)' 
    },
    'patora': {
        'model_path': 'model_patora.h5',
        'scaler_path': 'scaler_patora.gz',
        'csv_path': 'arecanut.csv', 
        'price_column': 'Modal Price (Rs./Quintal)'
    }
}

models = {}
scalers = {}

# --- Step 3: Load All Models and Scalers on Startup ---
def load_all_models():
    """Loads all models and scalers defined in the CONFIG."""
    all_loaded = True
    for key, paths in CONFIG.items():
        model_path = paths['model_path']
        scaler_path = paths['scaler_path']

        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                print(f"Loading model and scaler for '{key}'...")
                # Suppress TF messages for cleaner output during loading
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
                models[key] = load_model(model_path, compile=False) 
                scalers[key] = joblib.load(scaler_path)
                print(f"Successfully loaded '{key}'.")
            else:
                raise FileNotFoundError(f"Missing one or both files: '{model_path}' and '{scaler_path}'")
        except Exception as e:
            print(f"FATAL ERROR: Failed to load model or scaler for '{key}'. Error: {e}")
            all_loaded = False
    return all_loaded

# --- Step 4: Define Constants ---
SEQUENCE_LENGTH = 30 # Number of past days used for a single prediction
FORECAST_DAYS = 7 # Number of days to forecast

# --- Step 5: Function to Get Historical Data (Updated to return last date) ---
def get_recent_data(arecanut_type, days_for_plot):
    """
    Fetches the necessary data:
    1. Last SEQUENCE_LENGTH prices for LSTM input.
    2. Last 'days_for_plot' prices and dates for the historical chart part.
    3. The last date string for calculating future dates.
    """
    type_config = CONFIG.get(arecanut_type, {})
    csv_path = type_config.get('csv_path')
    price_column = type_config.get('price_column')

    if not csv_path or not os.path.exists(csv_path):
        return None, None, None, None, f"Data file '{csv_path}' not found."
    if not price_column:
        return None, None, None, None, f"Price column not configured for type '{arecanut_type}'."

    try:
        df = pd.read_csv(csv_path, parse_dates=['Price Date'])
        
        if price_column not in df.columns:
            return None, None, None, None, f"Column '{price_column}' not found in '{csv_path}'."

        df.rename(columns={'Price Date': 'date', price_column: 'price'}, inplace=True)
        
        # Total data points required for the entire sequence + plot history
        required_data_points = SEQUENCE_LENGTH + days_for_plot
        
        df_sorted = df.sort_values('date', ascending=True).tail(required_data_points)
        
        if len(df_sorted) < required_data_points:
             return None, None, None, None, f"Not enough data. Need at least {required_data_points} records, but found {len(df_sorted)}."

        # 1. Data for the LSTM input (last SEQUENCE_LENGTH prices)
        lstm_input_data = df_sorted['price'].values[-SEQUENCE_LENGTH:]
        
        # 2. Data for the historical graph plot (last 'days_for_plot' prices and dates)
        historical_plot_data = df_sorted[['date', 'price']].tail(days_for_plot)
        historical_prices = historical_plot_data['price'].values
        historical_dates = historical_plot_data['date'].dt.strftime('%b %d').tolist()

        # 3. The last date string for calculating future dates
        last_historical_date_str = historical_plot_data['date'].iloc[-1].strftime('%Y-%m-%d')


        return lstm_input_data, historical_prices, historical_dates, last_historical_date_str, None
    except Exception as e:
        print(f"Error reading data: {e}")
        return None, None, None, None, f"Error reading data for '{arecanut_type}': {e}"


# --- Step 5.1: Function for Multi-Step (7-Day) Prediction ---
def predict_next_7_days(model, scaler, past_prices, sequence_length, forecast_days):
    """
    Generates a multi-step price forecast by feeding the prediction back as input.
    """
    predicted_prices_7_days = []
    
    # 1. Scale the input data
    scaled_input = scaler.transform(past_prices.reshape(-1, 1)) 
    current_sequence = scaled_input.flatten().tolist() 

    for _ in range(forecast_days):
        # 2. Prepare input for the model
        input_data = np.array(current_sequence[-sequence_length:]).reshape(1, sequence_length, 1)

        # 3. Make a prediction (verbose=0 to suppress output)
        predicted_scaled_price = model.predict(input_data, verbose=0)
        
        # 4. Inverse transform the prediction to get the actual price
        predicted_price = scaler.inverse_transform(predicted_scaled_price)[0][0]

        # 5. Store the actual predicted price, rounded to 2 decimal places
        predicted_prices_7_days.append(round(float(predicted_price), 2))

        # 6. Update the sequence for the next prediction: append the new scaled prediction
        current_sequence.append(predicted_scaled_price[0][0]) 

    return predicted_prices_7_days

# --- Step 6: Define the Single-Day Prediction Endpoint (Updated signature handling) ---
@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Invalid request: payload must be in JSON format.'}), 400

    data = request.get_json()
    arecanut_type = data.get('type')

    if not arecanut_type or arecanut_type not in models:
        return jsonify({'error': f"Invalid or missing arecanut type '{arecanut_type}'."}), 400

    model = models[arecanut_type]
    scaler = scalers[arecanut_type]

    # Use SEQUENCE_LENGTH as the number of historical days needed for the input, but 
    # we only care about the past_prices array here. The extra returns are ignored (_).
    past_prices, _, _, _, error_message = get_recent_data(arecanut_type, days_for_plot=1) 
    if error_message:
        return jsonify({'error': error_message}), 400

    try:
        past_prices_reshaped = past_prices.reshape(-1, 1)
        scaled_prices = scaler.transform(past_prices_reshaped)
        input_data = scaled_prices.reshape(1, SEQUENCE_LENGTH, 1)
        predicted_scaled_price = model.predict(input_data, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_scaled_price)

        return jsonify({
            'type': arecanut_type,
            'predicted_price': round(float(predicted_price[0][0]), 2)
        })

    except Exception as e:
        print(f"Prediction error for type '{arecanut_type}': {e}")
        return jsonify({'error': 'An error occurred during single-day prediction.'}), 500

# --- Step 6.1: Define the Historical + 7-Day Forecast Endpoint (Fixed date calculation) ---
@app.route('/historical_and_forecast_data', methods=['POST'])
def historical_and_forecast_data():
    if not request.is_json:
        return jsonify({'error': 'Invalid request: payload must be in JSON format.'}), 400

    data = request.get_json()
    arecanut_type = data.get('type')

    if not arecanut_type or arecanut_type not in models:
        return jsonify({'error': f"Invalid or missing arecanut type '{arecanut_type}'."}), 400

    model = models[arecanut_type]
    scaler = scalers[arecanut_type]

    # Fetch the last 7 days of historical prices plus the 30 days needed for LSTM input
    lstm_input_data, historical_prices, historical_dates, last_date_str, error_message = get_recent_data(arecanut_type, days_for_plot=FORECAST_DAYS) 
    if error_message:
        return jsonify({'error': error_message}), 400
        
    try:
        # Generate the 7-day forecast
        forecast_prices = predict_next_7_days(model, scaler, lstm_input_data, SEQUENCE_LENGTH, FORECAST_DAYS)

        # Generate the future dates for the forecast part of the graph (FIXED)
        last_historical_date = pd.to_datetime(last_date_str)
        
        forecast_dates = []
        # Start calculating dates from the day after the last historical date
        start_date = last_historical_date + pd.Timedelta(days=1)
        for i in range(FORECAST_DAYS):
            future_date = start_date + pd.Timedelta(days=i)
            forecast_dates.append(future_date.strftime('%b %d'))

        # Prepare the combined data for the frontend
        return jsonify({
            'type': arecanut_type,
            'historical_prices': historical_prices.tolist(), 
            'historical_dates': historical_dates,
            'forecast_prices': forecast_prices,
            'forecast_dates': forecast_dates
        })

    except Exception as e:
        print(f"Historical/Forecast data error for type '{arecanut_type}': {e}")
        return jsonify({'error': f'An error occurred during data retrieval/forecast: {str(e)}'}), 500

# --- Step 7: Run the App ---
if __name__ == '__main__':
    # Load all models and scalers before starting the server
    if load_all_models():
        print("All models loaded. Starting Flask server...")
        # Use port 5001 as required by the frontend logic
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("Server could not start because some models failed to load.")