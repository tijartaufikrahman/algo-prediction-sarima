from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = FastAPI(
    title="SARIMA Forecast API",
    description="API Prediksi Harga (SARIMA) berbasis Python",
    version="1.0"
)

# =====================================================
# 🔥 CORS (INI YANG BIKIN FETCH JS BISA)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # boleh diganti domain kamu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# REQUEST BODY
# ===============================
class SarimaRequest(BaseModel):
    data: List[float]          # 84 data bulanan
    order: List[int]           # [p,d,q]
    seasonal_order: List[int]  # [P,D,Q,s]
    steps: int                 # jumlah bulan forecast

# ===============================
# RESPONSE
# ===============================
class SarimaResponse(BaseModel):
    forecast: List[float]

# ===============================
# ENDPOINT FORECAST
# ===============================
@app.post("/forecast", response_model=SarimaResponse)
def forecast_sarima(req: SarimaRequest):

    # ===== VALIDASI =====
    if len(req.data) < 24:
        raise HTTPException(
            status_code=400,
            detail="Data minimal 24 bulan"
        )

    if len(req.order) != 3:
        raise HTTPException(
            status_code=400,
            detail="order harus [p,d,q]"
        )

    if len(req.seasonal_order) != 4:
        raise HTTPException(
            status_code=400,
            detail="seasonal_order harus [P,D,Q,s]"
        )

    try:
        series = np.array(req.data, dtype=float)

        # ===== MODEL SARIMA (BENAR & STABIL) =====
        model = SARIMAX(
            series,
            order=tuple(req.order),
            seasonal_order=tuple(req.seasonal_order),
            simple_differencing=False,      # 🔥 WAJIB
            enforce_stationarity=False,     # 🔥 BIAR AMAN
            enforce_invertibility=False     # 🔥 BIAR AMAN
        )

        result = model.fit(disp=False)

        # ===== FORECAST =====
        forecast = result.forecast(steps=req.steps)

        return {
            "forecast": forecast.round(2).tolist()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SARIMA error: {str(e)}"
        )

# ===============================
# ROOT CHECK
# ===============================
@app.get("/")
def root():
    return {"status": "SARIMA API is running"}
