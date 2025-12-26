from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

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
    data: List[float]
    max_lag: int


class SarimaEvalRequest(BaseModel):
    data: List[float]
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

class ResidualDiagnosticRequest(BaseModel):
    residuals: List[float]



# ======================================================
# HELPER
# ======================================================
def to_float(x):
    return float(x) if x is not None else None


def to_bool(x):
    return bool(x)


# ======================================================
# ACF & PACF
# ======================================================
@app.post("/acf-pacf")
def calculate_acf_pacf(req: ACFPACFRequest):

    data = np.array(req.data, dtype=float)
    N = len(data)

    if N < req.max_lag + 1:
        return {"error": "Data terlalu sedikit"}

    limit = float(1.96 / np.sqrt(N))

    acf_vals = acf(data, nlags=req.max_lag)
    pacf_vals = pacf(data, nlags=req.max_lag, method="ywm")

    result = []
    for lag in range(1, req.max_lag + 1):
        result.append({
            "lag": lag,
            "acf": to_float(acf_vals[lag]),
            "acf_significant": to_bool(abs(acf_vals[lag]) > limit),
            "pacf": to_float(pacf_vals[lag]),
            "pacf_significant": to_bool(abs(pacf_vals[lag]) > limit)
        })

    return {
        "total_data": N,
        "significance_limit": limit,
        "result": result
    }


# ======================================================
# SARIMA EVALUATION (GRID SEARCH + RESIDUAL)
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

                    except Exception:
                        continue

    if not results:
        return {
            "error": "Semua kombinasi SARIMA gagal",
            "total_model": 0
        }

    # ===============================
    # PILIH MODEL TERBAIK
    # ===============================
    results.sort(key=lambda x: x["AIC"])
    best = results[0]

    # ===============================
    # FIT ULANG MODEL TERBAIK
    # ===============================
    model_best = SARIMAX(
        y,
        order=(best["p"], best["d"], best["q"]),
        seasonal_order=(best["P"], best["D"], best["Q"], best["s"]),
        simple_differencing=False,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_best = model_best.fit(disp=False)
    residuals = fitted_best.resid.tolist()

    return {
        "total_model": len(results),
        "best_model": best,
        "residuals": residuals,
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



# ======================================================
# RESIDUAL DIAGNOSTIC (WHITE NOISE & NORMALITY)
# ======================================================
@app.post("/diagnostic-residual")
def diagnostic_residual(req: ResidualDiagnosticRequest):

    residuals = np.array(req.residuals, dtype=float)

    if len(residuals) < 20:
        return {
            "error": "Jumlah residual terlalu sedikit untuk uji statistik"
        }

    # ===============================
    # WHITE NOISE - Ljung Box
    # ===============================
    lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_pvalue = float(lb["lb_pvalue"].iloc[0])
    white_noise = lb_pvalue > 0.05

    # ===============================
    # NORMALITY - Shapiro Wilk
    # ===============================
    _, shapiro_pvalue = shapiro(residuals)
    normal = shapiro_pvalue > 0.05

    # ===============================
    # KESIMPULAN
    # ===============================
    if white_noise and normal:
        conclusion = "Residual bersifat white noise dan berdistribusi normal. Model SARIMA layak digunakan."
    elif white_noise:
        conclusion = "Residual bersifat white noise namun tidak sepenuhnya normal. Model masih layak untuk prediksi jangka pendek."
    else:
        conclusion = "Residual tidak bersifat white noise. Model SARIMA perlu diperbaiki."

    return {
        "white_noise_test": {
            "method": "Ljung-Box",
            "p_value": round(lb_pvalue, 6),
            "result": "LULUS" if white_noise else "TIDAK LULUS"
        },
        "normality_test": {
            "method": "Shapiro-Wilk",
            "p_value": round(shapiro_pvalue, 6),
            "result": "LULUS" if normal else "TIDAK LULUS"
        },
        "conclusion": conclusion
    }



# ======================================================
# ROOT
# ======================================================
@app.get("/")
def root():
    return {
        "status": "API is running",
        "endpoints": [
            "/acf-pacf",
            "/sarima-evaluate",
            "/diagnostic-residual",
            "/forecast"
        ]
    }
