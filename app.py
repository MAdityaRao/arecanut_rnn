import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf

# Force silence on warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    BASE_DIR = Path(__file__).resolve().parent
    MODELS = {
        'adike': {
            'model_file': 'model_adike.h5',
            'scaler_file': 'scaler_adike.gz',
            'price_column': 'Max Price (Rs./Quintal)'
        },
        'patora': {
            'model_file': 'model_patora.h5',
            'scaler_file': 'scaler_patora.gz',
            'price_column': 'Modal Price (Rs./Quintal)'
        }
    }
    CSV_FILE = BASE_DIR / 'arecanut.csv'
    SEQUENCE_LENGTH = 30

class ArecaPredictor:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.df = self._load_data()
        self._load_models()

    def _load_data(self):
        if not self.config.CSV_FILE.exists():
            logger.error("arecanut.csv not found!")
            return None
        df = pd.read_csv(self.config.CSV_FILE)
        df['Price Date'] = pd.to_datetime(df['Price Date'])
        return df.sort_values('Price Date')

    def _load_models(self):
        """Uses the successful compile=False fallback strategy confirmed by your logs"""
        for key, cfg in self.config.MODELS.items():
            m_path = self.config.BASE_DIR / cfg['model_file']
            s_path = self.config.BASE_DIR / cfg['scaler_file']
            
            if m_path.exists() and s_path.exists():
                try:
                    # Using compile=False bypasses the Keras 3 metadata errors
                    self.models[key] = tf.keras.models.load_model(str(m_path), compile=False)
                    self.scalers[key] = joblib.load(str(s_path))
                    logger.info(f"Initialized {key} successfully.")
                except Exception as e:
                    logger.error(f"Failed to load {key}: {e}")

    def predict_7_days(self, model_key):
        if model_key not in self.models:
            return None, f"Model {model_key} not loaded"
        
        col = self.config.MODELS[model_key]['price_column']
        # Extract last 30 days for LSTM input
        data = self.df[col].tail(self.config.SEQUENCE_LENGTH).values.reshape(-1, 1)
        scaler = self.scalers[model_key]
        model = self.models[model_key]
        
        preds = []
        current_seq = data.copy()
        
        # Recursive prediction loop
        for _ in range(7):
            scaled = scaler.transform(current_seq[-30:])
            out = model.predict(scaled.reshape(1, 30, 1), verbose=0)
            price = scaler.inverse_transform(out)[0][0]
            preds.append(round(float(price), 2))
            # Append prediction to sequence for next step
            current_seq = np.append(current_seq, [[price]], axis=0)
            
        return preds, None

app = Flask(__name__)
CORS(app)
predictor = ArecaPredictor(Config())

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "models": list(predictor.models.keys())})

@app.route('/models')
def get_models():
    return jsonify({"models": list(predictor.models.keys()), "config": {"sequence_length": 30}})

@app.route('/predict', methods=['POST'])
def predict():
    key = request.json.get('type')
    preds, err = predictor.predict_7_days(key)
    if err: return jsonify({"error": err}), 400
    
    last_date = predictor.df['Price Date'].iloc[-1]
    return jsonify({
        "success": True, 
        "data": {
            "price": preds[0], 
            "date": (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        }
    })

@app.route('/forecast', methods=['POST'])
def forecast():
    key = request.json.get('type')
    preds, err = predictor.predict_7_days(key)
    if err: return jsonify({"error": err}), 400
    
    last_date = predictor.df['Price Date'].iloc[-1]
    dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]
    
    return jsonify({
        "success": True,
        "data": {
            "historical": {
                "prices": predictor.df[predictor.config.MODELS[key]['price_column']].tail(7).tolist(), 
                "dates": predictor.df['Price Date'].tail(7).dt.strftime('%Y-%m-%d').tolist()
            },
            "forecast": {"prices": preds, "dates": dates}
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)