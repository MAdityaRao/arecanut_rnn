
# Arecanut Price Predictor (Simple RNN)

This project implements a Recurrent Neural Network (RNN) to predict the future prices of arecanut varieties in Karnataka markets using historical data from `arecanut.csv`. Unlike standard feed-forward networks, this model uses recurrent connections to process sequences of market data.

## 🚀 Project Overview

1. **Data Processing**: The model utilizes a -day sliding window of historical prices. Data is normalized using a `MinMaxScaler` to improve training stability.
2. **Model Architecture**: A Sequential model utilizing **SimpleRNN** layers. It is designed to capture short-term dependencies in price movements before passing the data to a Dense output layer.
3. **Inference Engine**: A high-performance **FastAPI** backend that handles real-time prediction requests by processing the latest  entries from the dataset.
4. **User Interface**: A responsive, dark-themed dashboard that allows users to toggle between "Adike" and "Patora" varieties for instant predictions.

## 🛠️ Tech Stack

* **Backend**: FastAPI & Uvicorn (ASGI)
* **Deep Learning**: TensorFlow / Keras
* **Data Science**: Pandas, NumPy, Scikit-Learn
* **Frontend**: HTML5, CSS3 (Modern Glassmorphism), Vanilla JavaScript

## 📦 Installation

Ensure you have your virtual environment activated.

```bash
# Install required libraries
pip install fastapi uvicorn tensorflow pandas numpy scikit-learn joblib

```

## 🏃 Getting Started

### 1. Prepare Models

Ensure your trained Simple RNN models (`model_adike.keras`, `model_patora.keras`) and scalers are in the project root.

### 2. Start the Backend

```bash
python app.py

```

The API will initialize and load the CSV and Keras models into memory.

### 3. Open the Dashboard

Launch `index.html` in your browser. Ensure the frontend `API` constant points to `http://127.0.0.1:5001` to communicate with the FastAPI server.

## 📁 File Structure

* `app.py`: FastAPI application and prediction logic.
* `index.html`: Web interface.
* `arecanut.csv`: Historical price dataset.
* `model_*.keras`: Saved SimpleRNN models.
* `scaler_*.gz`: Saved MinMaxScaler objects.