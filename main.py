from fastapi import FastAPI, HTTPException 
import numpy as np 
import pandas as pd
import joblib
from pydantic import BaseModel, Field, validator
from typing import Literal
from enum import Enum


model = joblib.load('loan_pipeline_model.pkl')

app = FastAPI(title = 'Loan Approval Prediction')


class LoanRequest(BaseModel):

    no_of_dependents: int = Field(..., ge=0, le=10, example=2)
    education: Literal["Graduate", "Not Graduate"] = Field(..., example="Graduate")
    self_employed: Literal["Yes", "No"] = Field(..., example="No")
    income_annum: float = Field(..., ge=0, example=500000)
    loan_amount: float = Field(..., ge=0, example=200000)
    loan_term: int = Field(..., ge=6, le=360, example=120)
    cibil_score: int = Field(..., ge=300, le=900, example=750)
    Movable_assets: float = Field(..., ge=0, example=150000)
    Immovable_assets: float = Field(..., ge=0, example=500000)


@app.get('/')
def home():
    return {'messege' : "Loan Approval Prediction API is running 🚀"}


@app.post('/predict')
def prediction(data:LoanRequest):
    
    try:
        df = pd.DataFrame([data.dict()])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]


        return {
            "loan_status": "Approved" if prediction == 1 else "Rejected",
            "approval_probability": round(float(probability), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
