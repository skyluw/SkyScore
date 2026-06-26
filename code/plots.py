"""
SkyScore — gráficos de la app (matplotlib puro, sin dependencia de Streamlit).
Identidad visual: panel-instrumento. Acento marca naranja #FF6A00.
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import matplotlib.font_manager as fm

INK = "#14181F"; MUTED = "#6B7280"
NARANJA = "#FF6A00"; AZUL = "#1E40AF"
VERDE = "#16A34A"; AMBAR = "#F59E0B"; ROJO = "#E5484D"

plt.rcParams.update({"font.family": "DejaVu Sans"})  # fallback seguro

def risk_color(prob):
    if prob >= 0.5: return ROJO
    if prob >= 0.3: return AMBAR
    return VERDE

def verdict(prob):
    if prob >= 0.5: return "ALTO RIESGO", ROJO
    if prob >= 0.3: return "RIESGO MODERADO", AMBAR
    return "BAJO RIESGO", VERDE

def gauge(prob):
    """Medidor semicircular con zonas verde/ámbar/rojo y aguja en el valor."""
    fig, ax = plt.subplots(figsize=(5.0, 2.95))
    fig.patch.set_alpha(0)
    ax.set_aspect("equal"); ax.axis("off")
    R, W = 1.0, 0.26
    zonas = [(0.0, 0.30, VERDE), (0.30, 0.50, AMBAR), (0.50, 1.0, ROJO)]
    for lo, hi, c in zonas:
        ax.add_patch(Wedge((0, 0), R, 180*(1-hi), 180*(1-lo),
                           width=W, facecolor=c, edgecolor="white", linewidth=2.5))
    # aguja
    ang = math.radians(180*(1-prob))
    ax.plot([0, 0.80*math.cos(ang)], [0, 0.80*math.sin(ang)],
            color=INK, lw=3.2, solid_capstyle="round", zorder=5)
    ax.add_patch(Circle((0, 0), 0.055, color=INK, zorder=6))
    # etiquetas extremos
    ax.text(-1.02, -0.04, "0%", color=MUTED, fontsize=9, ha="center")
    ax.text(1.02, -0.04, "100%", color=MUTED, fontsize=9, ha="center")
    ax.set_xlim(-1.18, 1.18); ax.set_ylim(-0.16, 1.12)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig

def explanation_chart(top, label_fn):
    """Barras horizontales de contribuciones SHAP (naranja=empuja a incumplir)."""
    top = top[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    colors = [NARANJA if v > 0 else AZUL for v in top.values]
    ax.barh([label_fn(c) for c in top.index], top.values, color=colors, height=0.58)
    ax.axvline(0, color="#C9CDD4", lw=1)
    ax.margins(y=0.04)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#E6E8EC")
    ax.tick_params(length=0, labelsize=11.5, colors=INK, pad=6)
    ax.set_xticks([])
    ax.set_xlabel("←  reduce el riesgo            aumenta el riesgo  →",
                  fontsize=10.5, color=MUTED, labelpad=12)
    fig.subplots_adjust(left=0.34, right=0.97, top=0.98, bottom=0.16)
    return fig
