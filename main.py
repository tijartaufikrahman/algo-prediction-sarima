from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import warnings

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ======================================================
# APP INIT
# ======================================================
app = FastAPI(
    title="Time Series SARIMA API",
    description="ACF, PACF, Evaluasi & Forecast SARIMA (JSON Safe)",
    version="2.0"
)

# ======================================================
# CORS CONFIG
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# REQUEST MODELS
# ======================================================
class ACFPACFRequest(BaseModel):
    data: List[float]     # DATA SETELAH differencing FINAL
    max_lag: int


class SarimaEvalRequest(BaseModel):
    data: List[float]     # DATA ASLI (BELUM differencing)
    d: int
    D: int
    s: int

    max_p: int
    max_q: int
    max_P: int
    max_Q: int


class SarimaForecastRequest(BaseModel):
    data: List[float]
    order: List[int]            # [p,d,q]
    seasonal_order: List[int]   # [P,D,Q,s]
    steps: int


# ======================================================
# HELPER (JSON SAFE)
# ======================================================
def to_float(x):
    return float(x) if x is not None else None


def to_bool(x):
    return bool(x)


# ======================================================
# ACF & PACF ENDPOINT
# ======================================================
@app.post("/acf-pacf")
def calculate_acf_pacf(req: ACFPACFRequest):

    data = np.array(req.data, dtype=float)
    N = len(data)

    if N < req.max_lag + 1:
        return {
            "error": "Jumlah data terlalu sedikit",
            "total_data": N
        }

    # batas signifikansi 95%
    significance_limit = float(1.96 / np.sqrt(N))

    acf_values = acf(data, nlags=req.max_lag)
    pacf_values = pacf(data, nlags=req.max_lag, method="ywm")

    result = []
    for lag in range(1, req.max_lag + 1):
        result.append({
            "lag": lag,
            "acf": to_float(acf_values[lag]),
            "acf_significant": to_bool(abs(acf_values[lag]) > significance_limit),
            "pacf": to_float(pacf_values[lag]),
            "pacf_significant": to_bool(abs(pacf_values[lag]) > significance_limit)
        })

    return {
        "total_data": N,
        "significance_limit": significance_limit,
        "result": result
    }


# ======================================================
# SARIMA EVALUATION (GRID SEARCH)
# ======================================================
@app.post("/sarima-evaluate")
def sarima_evaluate(req: SarimaEvalRequest):

    y = np.array(req.data, dtype=float)

    min_len = max(30, req.s * 3)
    if len(y) < min_len:
        return {
            "error": f"Data terlalu sedikit (minimal {min_len})",
            "total_data": len(y)
        }

    results = []

    for p in range(req.max_p + 1):
        for q in range(req.max_q + 1):
            for P in range(req.max_P + 1):
                for Q in range(req.max_Q + 1):

                    if p == 0 and q == 0 and P == 0 and Q == 0:
                        continue

                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")

                            model = SARIMAX(
                                y,
                                order=(p, req.d, q),
                                seasonal_order=(P, req.D, Q, req.s),
                                simple_differencing=False,
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )

                            fitted = model.fit(disp=False)

                            resid = fitted.resid
                            rmse = float(np.sqrt(np.mean(resid ** 2)))

                            results.append({
                                "p": p,
                                "d": req.d,
                                "q": q,
                                "P": P,
                                "D": req.D,
                                "Q": Q,
                                "s": req.s,
                                "AIC": to_float(fitted.aic),
                                "RMSE": rmse
                            })

                    except Exception as e:
                        print(f"GAGAL SARIMA({p},{q},{P},{Q}) -> {e}")
                        continue

    if not results:
        return {
            "error": "Semua kombinasi SARIMA gagal",
            "total_model": 0
        }

    results.sort(key=lambda x: x["AIC"])

    return {
        "total_model": len(results),
        "best_model": results[0],
        "all_models": results
    }


# ======================================================
# SARIMA FORECAST
# ======================================================
@app.post("/forecast")
def forecast_sarima(req: SarimaForecastRequest):

    if len(req.data) < 24:
        raise HTTPException(
            status_code=400,
            detail="Data minimal 24 observasi"
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

        model = SARIMAX(
            series,
            order=tuple(req.order),
            seasonal_order=tuple(req.seasonal_order),
            simple_differencing=False,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit(disp=False)

        forecast = fitted.forecast(steps=req.steps)

        return {
            "forecast": forecast.round(2).tolist()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SARIMA error: {str(e)}"
        )


# ======================================================
# ROOT CHECK
# ======================================================
@app.get("/")
def root():
    return {
        "status": "API is running",
        "endpoints": [
            "/acf-pacf",
            "/sarima-evaluate",
            "/forecast"
        ]
    }
