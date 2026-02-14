"""
Arecanut Price Predictor API
Production-ready Flask application for predicting arecanut prices
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    """Application configuration"""
    
    # Model configurations
    MODELS = {
        'adike': {
            'model_file': 'model_adike.h5',
            'scaler_file': 'scaler_adike.gz',
            'price_column': 'Max Price (Rs./Quintal)',
            'display_name': 'Adike'
        },
        'patora': {
            'model_file': 'model_patora.h5',
            'scaler_file': 'scaler_patora.gz',
            'price_column': 'Modal Price (Rs./Quintal)',
            'display_name': 'Patora'
        }
    }
    
    # Data configuration
    CSV_FILE = 'arecanut.csv'
    SEQUENCE_LENGTH = 30  # Days of history needed for prediction
    FORECAST_DAYS = 7      # Days to forecast
    
    # API configuration
    PORT = 5001
    HOST = '0.0.0.0'
    DEBUG = False
    
    # File paths
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR
    DATA_FILE = BASE_DIR / CSV_FILE


# --- Model Manager ---
class ModelManager:
    """Manages loading and caching of ML models"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self._load_models()
    
    def _load_models(self):
        """Load all models and scalers"""
        logger.info("Loading models and scalers...")
        
        for model_key, model_config in self.config.MODELS.items():
            try:
                model_path = self.config.MODEL_DIR / model_config['model_file']
                scaler_path = self.config.MODEL_DIR / model_config['scaler_file']
                
                if not model_path.exists():
                    logger.warning(f"Model file not found: {model_path}")
                    continue
                    
                if not scaler_path.exists():
                    logger.warning(f"Scaler file not found: {scaler_path}")
                    continue
                
                # Suppress TensorFlow warnings
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                
                self.models[model_key] = load_model(str(model_path), compile=False)
                self.scalers[model_key] = joblib.load(str(scaler_path))
                
                logger.info(f"✓ Loaded model: {model_key}")
                
            except Exception as e:
                logger.error(f"✗ Failed to load model {model_key}: {str(e)}")
        
        if not self.models:
            logger.error("No models were loaded successfully!")
    
    def is_model_available(self, model_key):
        """Check if a specific model is available"""
        return model_key in self.models and model_key in self.scalers
    
    def get_model(self, model_key):
        """Get model by key"""
        return self.models.get(model_key)
    
    def get_scaler(self, model_key):
        """Get scaler by key"""
        return self.scalers.get(model_key)
    
    def get_available_models(self):
        """Get list of available models"""
        return list(self.models.keys())


# --- Data Manager ---
class DataManager:
    """Manages data loading and preprocessing"""
    
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        self.data_cache = {}
        self._load_data()
    
    def _load_data(self):
        """Load and cache CSV data"""
        if not self.config.DATA_FILE.exists():
            logger.error(f"Data file not found: {self.config.DATA_FILE}")
            return
        
        try:
            df = pd.read_csv(self.config.DATA_FILE)
            df['Price Date'] = pd.to_datetime(df['Price Date'])
            df.sort_values('Price Date', inplace=True)
            
            # Cache data for each model type
            for model_key, model_config in self.config.MODELS.items():
                if model_config['price_column'] in df.columns:
                    model_df = df[['Price Date', model_config['price_column']]].copy()
                    model_df.rename(columns={
                        'Price Date': 'date',
                        model_config['price_column']: 'price'
                    }, inplace=True)
                    self.data_cache[model_key] = model_df
                    logger.info(f"✓ Loaded data for: {model_key}")
                else:
                    logger.warning(f"Column {model_config['price_column']} not found for {model_key}")
                    
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
    
    def get_historical_data(self, model_key, days_needed):
        """
        Get historical data for prediction
        
        Returns:
            tuple: (prices_array, dates_list, last_date, error_message)
        """
        if model_key not in self.data_cache:
            return None, None, None, f"No data available for {model_key}"
        
        df = self.data_cache[model_key]
        
        # Calculate required data points
        required_points = self.config.SEQUENCE_LENGTH + days_needed
        
        if len(df) < required_points:
            return None, None, None, f"Insufficient data. Need {required_points} records, have {len(df)}"
        
        # Get last N records
        recent_df = df.tail(required_points).copy()
        
        # Extract data
        lstm_input = recent_df['price'].values[-self.config.SEQUENCE_LENGTH:]
        
        historical = recent_df.tail(days_needed)
        historical_prices = historical['price'].values
        historical_dates = historical['date'].dt.strftime('%Y-%m-%d').tolist()
        
        last_date = historical['date'].iloc[-1]
        
        return lstm_input, historical_prices, historical_dates, last_date, None
    
    def validate_data(self):
        """Validate that all necessary data is available"""
        if not self.data_cache:
            return False, "No data loaded"
        
        for model_key in self.config.MODELS.keys():
            if model_key not in self.data_cache:
                return False, f"Missing data for {model_key}"
        
        return True, "Data validation passed"


# --- Predictor ---
class PricePredictor:
    """Handles price predictions"""
    
    def __init__(self, config, model_manager, data_manager):
        self.config = config
        self.model_manager = model_manager
        self.data_manager = data_manager
    
    def predict_single(self, model_key):
        """Predict single day price"""
        if not self.model_manager.is_model_available(model_key):
            return None, f"Model '{model_key}' not available"
        
        # Get historical data
        lstm_input, _, _, last_date, error = self.data_manager.get_historical_data(
            model_key, days_needed=1
        )
        
        if error:
            return None, error
        
        try:
            model = self.model_manager.get_model(model_key)
            scaler = self.model_manager.get_scaler(model_key)
            
            # Prepare input
            scaled_input = scaler.transform(lstm_input.reshape(-1, 1))
            model_input = scaled_input.reshape(1, self.config.SEQUENCE_LENGTH, 1)
            
            # Predict
            scaled_prediction = model.predict(model_input, verbose=0)
            prediction = scaler.inverse_transform(scaled_prediction)[0][0]
            
            # Calculate next business day
            next_date = self._get_next_business_day(last_date)
            
            return {
                'price': round(float(prediction), 2),
                'date': next_date.strftime('%Y-%m-%d'),
                'model': model_key
            }, None
            
        except Exception as e:
            logger.error(f"Prediction error for {model_key}: {str(e)}")
            return None, f"Prediction failed: {str(e)}"
    
    def predict_forecast(self, model_key):
        """Generate 7-day forecast"""
        if not self.model_manager.is_model_available(model_key):
            return None, f"Model '{model_key}' not available"
        
        # Get historical data
        lstm_input, hist_prices, hist_dates, last_date, error = self.data_manager.get_historical_data(
            model_key, days_needed=self.config.FORECAST_DAYS
        )
        
        if error:
            return None, error
        
        try:
            model = self.model_manager.get_model(model_key)
            scaler = self.model_manager.get_scaler(model_key)
            
            # Generate forecast
            forecast_prices = self._generate_forecast(
                model, scaler, lstm_input
            )
            
            # Generate forecast dates
            forecast_dates = []
            current_date = last_date
            for i in range(self.config.FORECAST_DAYS):
                current_date = self._get_next_business_day(current_date)
                forecast_dates.append(current_date.strftime('%Y-%m-%d'))
            
            return {
                'historical': {
                    'prices': [float(p) for p in hist_prices],
                    'dates': hist_dates
                },
                'forecast': {
                    'prices': [float(p) for p in forecast_prices],
                    'dates': forecast_dates
                },
                'model': model_key
            }, None
            
        except Exception as e:
            logger.error(f"Forecast error for {model_key}: {str(e)}")
            return None, f"Forecast failed: {str(e)}"
    
    def _generate_forecast(self, model, scaler, initial_sequence):
        """Generate multi-step forecast"""
        forecast = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(self.config.FORECAST_DAYS):
            # Scale current sequence
            scaled_seq = scaler.transform(current_sequence.reshape(-1, 1))
            model_input = scaled_seq.reshape(1, self.config.SEQUENCE_LENGTH, 1)
            
            # Predict next value
            scaled_pred = model.predict(model_input, verbose=0)
            pred_value = scaler.inverse_transform(scaled_pred)[0][0]
            forecast.append(round(float(pred_value), 2))
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], pred_value)
        
        return forecast
    
    @staticmethod
    def _get_next_business_day(date):
        """Get next business day (skip weekends)"""
        next_date = date + timedelta(days=1)
        while next_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            next_date += timedelta(days=1)
        return next_date


# --- Flask Application ---
def create_app():
    """Application factory"""
    app = Flask(__name__)
    CORS(app)
    
    # Initialize managers
    config = Config()
    model_manager = ModelManager(config)
    data_manager = DataManager(config, model_manager)
    predictor = PricePredictor(config, model_manager, data_manager)
    
    # Store managers in app config for access in routes
    app.config['MODEL_MANAGER'] = model_manager
    app.config['DATA_MANAGER'] = data_manager
    app.config['PREDICTOR'] = predictor
    app.config['APP_CONFIG'] = config
    
    # Validate setup
    data_valid, data_message = data_manager.validate_data()
    if not data_valid:
        logger.warning(f"Data validation: {data_message}")
    
    if not model_manager.get_available_models():
        logger.warning("No ML models available. Running in limited mode.")
    
    # --- Routes ---
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        model_manager = app.config['MODEL_MANAGER']
        data_manager = app.config['DATA_MANAGER']
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_available': model_manager.get_available_models(),
            'data_loaded': list(data_manager.data_cache.keys())
        })
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Single-day prediction endpoint"""
        predictor = app.config['PREDICTOR']
        
        if not request.is_json:
            return jsonify({'error': 'JSON payload required'}), 400
        
        data = request.get_json()
        model_key = data.get('type')
        
        if not model_key:
            return jsonify({'error': 'Missing model type'}), 400
        
        result, error = predictor.predict_single(model_key)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    @app.route('/forecast', methods=['POST'])
    def forecast():
        """7-day forecast endpoint"""
        predictor = app.config['PREDICTOR']
        
        if not request.is_json:
            return jsonify({'error': 'JSON payload required'}), 400
        
        data = request.get_json()
        model_key = data.get('type')
        
        if not model_key:
            return jsonify({'error': 'Missing model type'}), 400
        
        result, error = predictor.predict_forecast(model_key)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    @app.route('/models', methods=['GET'])
    def list_models():
        """List available models"""
        model_manager = app.config['MODEL_MANAGER']
        config = app.config['APP_CONFIG']
        
        return jsonify({
            'models': model_manager.get_available_models(),
            'config': {
                'sequence_length': config.SEQUENCE_LENGTH,
                'forecast_days': config.FORECAST_DAYS
            }
        })
    
    return app


# --- Main entry point ---
if __name__ == '__main__':
    app = create_app()
    
    # Get managers from app config
    model_manager = app.config['MODEL_MANAGER']
    config = app.config['APP_CONFIG']
    
    logger.info(f"Starting server on {config.HOST}:{config.PORT}")
    logger.info(f"Available models: {model_manager.get_available_models()}")
    
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )