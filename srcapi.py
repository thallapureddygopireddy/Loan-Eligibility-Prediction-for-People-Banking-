from fastapi import FastAPI, Query, HTTPException
from typing import List
import joblib, json, os, pandas as pd, numpy as np
from .schema import LoanApplication

MODEL_PATH = "model.pkl"
COLUMNS_PATH = "columns.json"

app = FastAPI(title="Loan Eligibility API")

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Model not found. Run train.py first!")
    model = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH) as f:
        cols = json.load(f)["columns"]
    return model, cols

model, columns = load_artifacts()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(applications: List[LoanApplication], threshold: float = Query(0.5)):
    try:
        df = pd.DataFrame([a.model_dump() for a in applications])
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        df = df[columns]
        scores = model.predict_proba(df)[:,1]
        decisions = (scores >= threshold).astype(int).tolist()
        return [{"eligible": bool(d), "probability": float(s)} for d, s in zip(decisions, scores)]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
