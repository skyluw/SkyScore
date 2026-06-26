"""
SkyScore — lógica de scoring reutilizable (la usa la app y los tests).
Construye el vector de features, predice la probabilidad y la explica con SHAP.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "skyscore_model.joblib"

LABELS = {
    "LIMIT_BAL": "Límite de crédito", "AGE": "Edad", "SEX": "Sexo",
    "EDUCATION": "Educación", "MARRIAGE": "Estado civil",
    "utilizacion": "Utilización del crédito", "meses_atraso": "Meses con atraso",
    "ratio_pago": "Ratio pago/deuda",
}
for i in range(1, 7):
    LABELS[f"PAY_{i}"] = f"Atraso mes {i}" + (" (último)" if i == 1 else "")
    LABELS[f"BILL_AMT{i}"] = f"Deuda mes {i}"
    LABELS[f"PAY_AMT{i}"] = f"Pago mes {i}"

def label(col):
    return LABELS.get(col, col)

_cache = {}

def load_model():
    if "bundle" not in _cache:
        _cache["bundle"] = joblib.load(MODEL_PATH)
        _cache["explainer"] = shap.TreeExplainer(_cache["bundle"]["model"])
    return _cache["bundle"], _cache["explainer"]

def build_features(limite, edad, sexo, educacion, estado_civil,
                   atrasos, deuda_mensual, pago_mensual):
    """
    atrasos: lista de 6 enteros (estado de pago de los últimos 6 meses, -2..8).
    deuda_mensual / pago_mensual: montos promedio (NT$) — se replican a los 6 meses.
    """
    row = {
        "LIMIT_BAL": limite, "SEX": sexo, "EDUCATION": educacion,
        "MARRIAGE": estado_civil, "AGE": edad,
    }
    for i in range(1, 7):
        row[f"PAY_{i}"] = atrasos[i - 1]
        row[f"BILL_AMT{i}"] = deuda_mensual
        row[f"PAY_AMT{i}"] = pago_mensual
    # features derivadas (idénticas al entrenamiento)
    util = (deuda_mensual / limite) if limite else 0.0
    row["utilizacion"] = float(np.clip(util, -1, 5))
    row["meses_atraso"] = int(sum(1 for a in atrasos if a >= 1))
    row["ratio_pago"] = (6 * pago_mensual) / (abs(6 * deuda_mensual) + 1)

    bundle, _ = load_model()
    return pd.DataFrame([row])[bundle["features"]]

def predict_proba(X):
    bundle, _ = load_model()
    return float(bundle["model"].predict_proba(X)[:, 1][0])

def explain(X, top_n=8):
    """Devuelve las contribuciones SHAP (clase incumple) ordenadas por magnitud."""
    bundle, explainer = load_model()
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    contrib = pd.Series(np.ravel(sv[0]), index=bundle["features"])
    return contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)

# Presets para la demo (calibrados para caer en cada zona de riesgo)
PRESET_BAJO = dict(limite=320000, edad=45, sexo=2, educacion=1, estado_civil=2,
                   atrasos=[-1, -1, -1, -1, -1, -1], deuda_mensual=8000, pago_mensual=12000)
PRESET_MODERADO = dict(limite=60000, edad=31, sexo=2, educacion=2, estado_civil=1,
                       atrasos=[0, 0, 0, 0, -1, -1], deuda_mensual=48000, pago_mensual=2000)
PRESET_ALTO = dict(limite=15000, edad=23, sexo=1, educacion=3, estado_civil=2,
                   atrasos=[3, 3, 2, 2, 2, 2], deuda_mensual=14500, pago_mensual=0)
