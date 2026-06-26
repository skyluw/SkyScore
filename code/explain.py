"""
SkyScore — Explicabilidad con SHAP.
Muestra (1) qué factores impulsan el riesgo a nivel global y
(2) por qué el modelo dio cierta predicción a un cliente concreto.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, shap

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/"reports"/"figures"; FIG.mkdir(parents=True, exist_ok=True)
AZUL, NARANJA = "#2563eb", "#ff6a00"
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})

# Nombres legibles para mostrar
LABELS = {
    "LIMIT_BAL":"Límite de crédito", "AGE":"Edad", "SEX":"Sexo",
    "EDUCATION":"Educación", "MARRIAGE":"Estado civil",
    "PAY_1":"Atraso mes 1 (último)", "PAY_2":"Atraso mes 2", "PAY_3":"Atraso mes 3",
    "PAY_4":"Atraso mes 4", "PAY_5":"Atraso mes 5", "PAY_6":"Atraso mes 6",
    "utilizacion":"Utilización del crédito", "meses_atraso":"Meses con atraso",
    "ratio_pago":"Ratio pago/deuda",
}
for i in range(1,7):
    LABELS[f"BILL_AMT{i}"] = f"Deuda mes {i}"
    LABELS[f"PAY_AMT{i}"]  = f"Pago mes {i}"
def lbl(c): return LABELS.get(c, c)

# ---------- replicar preprocesamiento del entrenamiento ----------
df = pd.read_csv(ROOT/"data"/"credit_default.csv")
df = df.rename(columns={"default.payment.next.month":"default","PAY_0":"PAY_1"}).drop(columns=["ID"])
df["EDUCATION"] = df["EDUCATION"].replace({0:4,5:4,6:4})
df["MARRIAGE"]  = df["MARRIAGE"].replace({0:3})
bill = [f"BILL_AMT{i}" for i in range(1,7)]; payc=[f"PAY_AMT{i}" for i in range(1,7)]
delay=[f"PAY_{i}" for i in range(1,7)]
df["utilizacion"] = (df[bill].mean(axis=1)/df["LIMIT_BAL"].replace(0,np.nan)).fillna(0).clip(-1,5)
df["meses_atraso"] = (df[delay]>=1).sum(axis=1)
df["ratio_pago"] = df[payc].sum(axis=1)/(df[bill].sum(axis=1).abs()+1)

bundle = joblib.load(ROOT/"models"/"skyscore_model.joblib")
model, FEATURES = bundle["model"], bundle["features"]
X = df[FEATURES]

# ---------- SHAP (TreeExplainer sobre el Random Forest) ----------
sample = X.sample(1500, random_state=8)
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(sample)
base = explainer.expected_value
# Seleccionar la clase 1 (incumple) de forma robusta entre versiones de shap
if isinstance(sv, list):
    sv = sv[1]; base = base[1] if np.ndim(base) else base
elif getattr(sv, "ndim", 2) == 3:
    sv = sv[:, :, 1]; base = np.ravel(base)[1]
base = float(np.ravel(base)[-1]) if np.ndim(base) else float(base)

# ---------- (1) Importancia global: media |SHAP| ----------
imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=True).tail(12)
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.barh([lbl(c) for c in imp.index], imp.values, color=AZUL)
ax.set_title("¿Qué factores impulsan el riesgo? (importancia global SHAP)")
ax.set_xlabel("Impacto promedio en la predicción"); ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(FIG/"08_shap_global.png"); plt.close()
print("Top factores globales:")
print(imp.sort_values(ascending=False).head(6).rename(index=lbl).round(4).to_string())

# ---------- (2) Explicación local de clientes concretos ----------
proba = model.predict_proba(sample)[:, 1]
idx_riesgo = sample.index[proba.argmax()]
idx_seguro = sample.index[proba.argmin()]

def explica_cliente(idx, nombre_archivo, titulo):
    pos = sample.index.get_loc(idx)
    contrib = pd.Series(sv[pos], index=FEATURES)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
    colors = [NARANJA if v > 0 else AZUL for v in top.values]
    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.barh([lbl(c) for c in top.index], top.values, color=colors)
    ax.axvline(0, color="#333", lw=0.8)
    p = model.predict_proba(X.loc[[idx]])[:, 1][0]
    ax.set_title(f"{titulo}\nProbabilidad de incumplimiento: {p:.0%}")
    ax.set_xlabel("← empuja a PAGA      |      empuja a INCUMPLE →")
    fig.tight_layout(); fig.savefig(FIG/nombre_archivo); plt.close()
    return p, top

p_r, _ = explica_cliente(idx_riesgo, "09_shap_cliente_riesgoso.png", "Cliente de ALTO riesgo")
p_s, _ = explica_cliente(idx_seguro, "10_shap_cliente_seguro.png", "Cliente de BAJO riesgo")
print(f"\nEjemplo riesgoso: {p_r:.0%}  |  Ejemplo seguro: {p_s:.0%}")
print("Gráficos SHAP guardados. Explicabilidad lista.")
