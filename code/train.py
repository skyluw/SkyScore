"""
SkyScore — Entrenamiento y evaluación de modelos de riesgo crediticio.
Compara: Regresión Logística (baseline) vs Random Forest vs Gradient Boosting.
Métricas pensadas para clases desbalanceadas (F1, recall, ROC-AUC), no accuracy.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, f1_score, recall_score, precision_score,
                             average_precision_score)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/"reports"/"figures"; FIG.mkdir(parents=True, exist_ok=True)
MODELS = ROOT/"models"; MODELS.mkdir(parents=True, exist_ok=True)
AZUL, NARANJA, VERDE, GRIS = "#2563eb", "#ff6a00", "#16a34a", "#9aa4b2"
plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

# ---------- 1. Cargar y limpiar ----------
df = pd.read_csv(ROOT/"data"/"credit_default.csv")
df = df.rename(columns={"default.payment.next.month": "default", "PAY_0": "PAY_1"})
df = df.drop(columns=["ID"])
# Consolidar categorías fuera de catálogo en "otros/desconocido"
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})   # -> 4 = otros
df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})                 # -> 3 = otros

# ---------- 2. Feature engineering ligero (ayuda al modelo y a la explicación) ----------
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
pay_cols  = [f"PAY_AMT{i}" for i in range(1, 7)]
delay_cols = [f"PAY_{i}" for i in range(1, 7)]
df["utilizacion"] = df[bill_cols].mean(axis=1) / df["LIMIT_BAL"].replace(0, np.nan)
df["utilizacion"] = df["utilizacion"].fillna(0).clip(-1, 5)
df["meses_atraso"] = (df[delay_cols] >= 1).sum(axis=1)          # cuántos meses estuvo atrasado
df["ratio_pago"]   = df[pay_cols].sum(axis=1) / (df[bill_cols].sum(axis=1).abs() + 1)

X = df.drop(columns=["default"])
y = df["default"]
FEATURES = list(X.columns)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=8)
print(f"Train: {X_tr.shape[0]:,}  |  Test: {X_te.shape[0]:,}  |  Features: {len(FEATURES)}")

# ---------- 3. Modelos ----------
modelos = {
    "Regresión Logística": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=8)),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=8),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.08, max_iter=400, random_state=8),
}

resultados, curvas, test_proba = {}, {}, {}
for nombre, modelo in modelos.items():
    if nombre == "Gradient Boosting":
        sw = compute_sample_weight("balanced", y_tr)
        modelo.fit(X_tr, y_tr, sample_weight=sw)
    else:
        modelo.fit(X_tr, y_tr)
    proba = modelo.predict_proba(X_te)[:, 1]
    test_proba[nombre] = proba   # predicciones sobre el test (sin fuga de datos)
    pred = (proba >= 0.5).astype(int)
    resultados[nombre] = {
        "precision": precision_score(y_te, pred),
        "recall":    recall_score(y_te, pred),
        "f1":        f1_score(y_te, pred),
        "roc_auc":   roc_auc_score(y_te, proba),
        "pr_auc":    average_precision_score(y_te, proba),
    }
    curvas[nombre] = roc_curve(y_te, proba)

tabla = pd.DataFrame(resultados).T[["precision", "recall", "f1", "roc_auc", "pr_auc"]]
print("\n=== Comparación de modelos (clase = incumple) ===")
print((tabla*100).round(1).to_string())

mejor = tabla["f1"].idxmax()
print(f"\nMejor modelo por F1: {mejor}")

# ---------- 4. Reentrenar el mejor en TODO y guardarlo ----------
best_model = modelos[mejor]
if mejor == "Gradient Boosting":
    best_model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
else:
    best_model.fit(X, y)
joblib.dump({"model": best_model, "features": FEATURES}, MODELS/"skyscore_model.joblib")

# ---------- 5. Gráficos de evaluación ----------
# (a) comparación de métricas
fig, ax = plt.subplots(figsize=(8.2, 4.4))
met = ["recall", "f1", "roc_auc"]; labels = ["Recall", "F1", "ROC-AUC"]
x = np.arange(len(met)); w = 0.26
for i, (nombre, col) in enumerate(zip(modelos, [GRIS, AZUL, NARANJA])):
    vals = [resultados[nombre][m] for m in met]
    ax.bar(x + (i-1)*w, vals, w, label=nombre, color=col)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0, 1); ax.set_title("Comparación de modelos (clase incumple)")
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"05_comparacion_modelos.png"); plt.close()

# (b) curvas ROC
fig, ax = plt.subplots(figsize=(5.6, 5))
for nombre, col in zip(modelos, [GRIS, AZUL, NARANJA]):
    fpr, tpr, _ = curvas[nombre]
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{nombre} (AUC {resultados[nombre]['roc_auc']:.2f})")
ax.plot([0,1],[0,1], ls="--", color="#cccccc")
ax.set_xlabel("Falsos positivos"); ax.set_ylabel("Verdaderos positivos")
ax.set_title("Curvas ROC"); ax.legend(fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(FIG/"06_curvas_roc.png"); plt.close()

# (c) matriz de confusión del mejor — usando las predicciones sobre el TEST no visto
cm = confusion_matrix(y_te, (test_proba[mejor] >= 0.5).astype(int))
fig, ax = plt.subplots(figsize=(4.8, 4.4))
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v:,}", ha="center", va="center",
            color="white" if v > cm.max()/2 else "black", fontsize=13)
ax.set_xticks([0,1]); ax.set_xticklabels(["Paga","Incumple"])
ax.set_yticks([0,1]); ax.set_yticklabels(["Paga","Incumple"])
ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusión — {mejor}"); ax.grid(False)
fig.tight_layout(); fig.savefig(FIG/"07_matriz_confusion.png"); plt.close()

json.dump({k: {m: round(v,4) for m,v in d.items()} for k,d in resultados.items()},
          open(MODELS/"metrics.json","w"), indent=2, ensure_ascii=False)
print("\nModelo guardado en models/skyscore_model.joblib")
print("Gráficos en reports/figures/. Entrenamiento completo.")
