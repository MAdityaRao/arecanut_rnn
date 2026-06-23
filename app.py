from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Resources ---
df = pd.read_csv("arecanut.csv")
loaded_models = {
    "adike": load_model("model_adike.keras", compile=False),
    "patora": load_model("model_patora.keras", compile=False)
}
loaded_scalers = {
    "adike": joblib.load("scaler_adike.gz"),
    "patora": joblib.load("scaler_patora.gz")
}

# --- Request Schema ---
class PredictRequest(BaseModel):
    type: str

# --- Routes ---
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        model_type = request.type

        if model_type not in loaded_models:
            raise HTTPException(status_code=400, detail="Invalid model type")

        model = loaded_models[model_type]
        scaler = loaded_scalers[model_type]

        col = "Max Price (Rs./Quintal)" if model_type == "adike" else "Modal Price (Rs./Quintal)"
        
        last_30 = df.dropna(subset=[col])[col].values[-30:].reshape(30, 1)
        scaled_data = scaler.transform(last_30)
        X = scaled_data.reshape(1, 30, 1)
        
        scaled_pred = model.predict(X, verbose=0)
        prediction = scaler.inverse_transform(scaled_pred)
        price = float(prediction[0][0])

        return {
            "success": True,
            "data": {
                "price": round(price, 2),
                "date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# THIS ROUTE MUST BE HERE TO FIX THE GRAPH 404 ERROR
@app.get("/history/{model_type}")
async def get_history(model_type: str):
    if model_type not in ["adike", "patora"]:
        raise HTTPException(status_code=400, detail="Invalid type")
    
    col = "Max Price (Rs./Quintal)" if model_type == "adike" else "Modal Price (Rs./Quintal)"
    last_15 = df.dropna(subset=[col])[col].values[-15:].tolist()
    return {"history": last_15}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)