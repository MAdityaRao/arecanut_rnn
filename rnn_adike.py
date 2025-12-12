import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import joblib

# --- 1. Load and Prepare Data ---
df = pd.read_csv("arecanut.csv")
df['Price Date'] = pd.to_datetime(df['Price Date'])
df.sort_values('Price Date', inplace=True)
price_data = df[['Max Price (Rs./Quintal)']].values

# --- 2. Scale the Data ---
# LSTMs work best with data scaled between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(price_data)

# --- 3. Create Training Sequences ---
# We'll use the last 30 days of data to predict the next day's price
SEQUENCE_LENGTH = 30
X, y = [], []
for i in range(len(scaled_data) - SEQUENCE_LENGTH):
    X.append(scaled_data[i:(i + SEQUENCE_LENGTH), 0])
    y.append(scaled_data[i + SEQUENCE_LENGTH, 0])

X, y = np.array(X), np.array(y)

# Reshape X to be [samples, timesteps, features] which is required for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))


# --- 4. Build and Train the LSTM Model ---
model = Sequential()
model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(SimpleRNN(units=50, return_sequences=False))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100, batch_size=32, verbose=1)

# --- 5. Save the Model and Scaler ---
model.save('model_adike.h5')
joblib.dump(scaler, 'scaler_adike.gz')
print(f"tommorow's price: {scaler.inverse_transform(model.predict(X[-1].reshape(1, SEQUENCE_LENGTH, 1)))[0][0]}")
print("\\nTraining complete. Model saved as 'model_adike.h5' and scaler as 'scaler_adike.gz'.")