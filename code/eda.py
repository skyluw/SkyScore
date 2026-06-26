"""
SkyScore — Análisis Exploratorio de Datos (EDA)
Dataset: Default of Credit Card Clients (UCI), 30,000 clientes de Taiwán (2005).
Objetivo: entender los datos antes de modelar la probabilidad de incumplimiento.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "credit_default.csv"
FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Paleta SkyScore
AZUL, NARANJA, GRIS = "#2563eb", "#ff6a00", "#9aa4b2"
plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})

df = pd.read_csv(DATA)
df = df.rename(columns={"default.payment.next.month": "default", "PAY_0": "PAY_1"})
TARGET = "default"

print("="*60)
print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")
print(f"Valores faltantes (NaN): {int(df.isna().sum().sum())}")
print(f"Filas duplicadas: {int(df.duplicated().sum())}")

rate = df[TARGET].mean()
print(f"\nTasa de incumplimiento global: {rate:.1%}  (clases desbalanceadas)")
print(df[TARGET].value_counts().rename({0: "Paga (0)", 1: "Incumple (1)"}).to_string())

# Anomalías documentadas en categóricas
print("\nCategorías fuera de catálogo (a limpiar luego):")
print("  EDUCATION:", sorted(df.EDUCATION.unique()), " (0,5,6 = desconocido)")
print("  MARRIAGE :", sorted(df.MARRIAGE.unique()), " (0 = desconocido)")

def default_rate_by(col, labels=None, order=None):
    g = df.groupby(col)[TARGET].agg(["mean", "size"])
    if order is not None:
        g = g.reindex(order)
    if labels:
        g.index = [labels.get(i, i) for i in g.index]
    return g

# ---------- Figura 1: balance de clases ----------
fig, ax = plt.subplots(figsize=(5.2, 4))
counts = df[TARGET].value_counts().sort_index()
ax.bar(["Paga", "Incumple"], counts.values, color=[GRIS, NARANJA])
for i, v in enumerate(counts.values):
    ax.text(i, v + 300, f"{v:,}\n({v/len(df):.0%})", ha="center", fontsize=10)
ax.set_title("Balance de clases: ¿quién incumple el mes siguiente?")
ax.set_ylabel("N° de clientes"); ax.set_ylim(0, counts.max()*1.15)
fig.tight_layout(); fig.savefig(FIG/"01_balance_clases.png"); plt.close()

# ---------- Figura 2: tasa de default por estado de pago más reciente (PAY_1) ----------
g = default_rate_by("PAY_1").sort_index()
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.bar(g.index.astype(str), g["mean"], color=AZUL)
ax.axhline(rate, color=NARANJA, ls="--", lw=1.5, label=f"Promedio global {rate:.0%}")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("Tasa de incumplimiento según el último estado de pago (PAY_1)")
ax.set_xlabel("PAY_1  (-1/0 = al día, ≥1 = meses de atraso)")
ax.set_ylabel("Tasa de incumplimiento"); ax.legend()
fig.tight_layout(); fig.savefig(FIG/"02_default_por_PAY1.png"); plt.close()
print("\nTasa de default por PAY_1 (atraso reciente):")
print((g["mean"]*100).round(1).to_string())

# ---------- Figura 3: tasa de default por límite de crédito ----------
df["limite_bin"] = pd.qcut(df.LIMIT_BAL, 6, duplicates="drop")
g3 = df.groupby("limite_bin")[TARGET].mean()
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.plot(range(len(g3)), g3.values, marker="o", color=AZUL, lw=2)
ax.axhline(rate, color=NARANJA, ls="--", lw=1.5, label=f"Promedio global {rate:.0%}")
ax.set_xticks(range(len(g3)))
ax.set_xticklabels([f"{int(iv.left/1000)}k-{int(iv.right/1000)}k" for iv in g3.index], rotation=20)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title("A mayor límite de crédito, menor incumplimiento")
ax.set_xlabel("Límite de crédito (NT$)"); ax.set_ylabel("Tasa de incumplimiento"); ax.legend()
fig.tight_layout(); fig.savefig(FIG/"03_default_por_limite.png"); plt.close()

# ---------- Figura 4: default por educación y por sexo ----------
edu_lbl = {1:"Posgrado",2:"Universidad",3:"Secundaria",4:"Otros",5:"Desc.",6:"Desc.",0:"Desc."}
sex_lbl = {1:"Hombre",2:"Mujer"}
ge = default_rate_by("EDUCATION", edu_lbl, order=[1,2,3,4])
gs = default_rate_by("SEX", sex_lbl, order=[1,2])
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
axes[0].bar(ge.index, ge["mean"], color=AZUL)
axes[0].axhline(rate, color=NARANJA, ls="--", lw=1.3)
axes[0].set_title("Por nivel educativo"); axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
axes[1].bar(gs.index, gs["mean"], color=AZUL)
axes[1].axhline(rate, color=NARANJA, ls="--", lw=1.3)
axes[1].set_title("Por sexo"); axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
fig.suptitle("Tasa de incumplimiento por perfil demográfico")
fig.tight_layout(); fig.savefig(FIG/"04_default_demografia.png"); plt.close()

print("\nFiguras guardadas en reports/figures/")
print("EDA completado.")
