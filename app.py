"""
SkyScore — App web (Streamlit).
Demo interactiva: perfil del cliente -> probabilidad de incumplimiento + por qué (SHAP).
Ejecutar:  python -m streamlit run app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

import streamlit as st
import scoring as s
import plots as p

st.set_page_config(page_title="SkyScore — Riesgo crediticio explicable",
                   page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, p, label, div { font-family: 'Inter', sans-serif; }
h1,h2,h3,h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing:-.02em; color:#14181F; }

/* ocultar cromo de streamlit */
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer { display:none !important; }
.block-container { padding-top: 1.4rem !important; padding-bottom: 2rem; max-width: 1180px; }

/* hero */
.hero { border-bottom: 1px solid #E6E8EC; padding-bottom: 1rem; margin-bottom: 1.4rem; }
.brand { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2.1rem; color:#14181F; line-height:1; }
.brand .dot { color:#FF6A00; }
.thesis { color:#6B7280; font-size:1.02rem; margin-top:.35rem; }

/* eyebrow */
.eyebrow { font-family:'Space Grotesk',sans-serif; font-size:.74rem; font-weight:600;
           letter-spacing:.16em; color:#FF6A00; text-transform:uppercase; margin-bottom:.2rem; }

/* readout */
.score { font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:3.4rem; line-height:1; }
.score-sub { color:#6B7280; font-size:.92rem; margin-top:-.2rem; }
.pill { display:inline-block; font-weight:600; font-size:.9rem; padding:.32rem .8rem;
        border-radius:999px; margin-top:.7rem; }

/* inputs un poco más suaves */
[data-baseweb="input"] > div, [data-baseweb="select"] > div { border-radius:10px !important; }
.stButton > button { border-radius:999px; border:1px solid #E6E8EC; font-weight:600;
                     padding:.4rem 1rem; transition:all .15s; }
.stButton > button:hover { border-color:#FF6A00; color:#FF6A00; }
hr { margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><div class="brand"> Sky<span class="dot">Score</span></div>'
    '<div class="thesis">Mide la probabilidad de que un cliente incumpla su próximo pago — y explica por qué.</div></div>',
    unsafe_allow_html=True)

# ---- presets (escriben el estado de TODOS los widgets) ----
def aplicar_preset(preset):
    st.session_state.limite = preset["limite"]
    st.session_state.edad = preset["edad"]
    st.session_state.sexo = preset["sexo"]
    st.session_state.educacion = preset["educacion"]
    st.session_state.estado = preset["estado_civil"]
    st.session_state.deuda = preset["deuda_mensual"]
    st.session_state.pago = preset["pago_mensual"]
    for i in range(6):
        st.session_state[f"pay{i}"] = preset["atrasos"][i]

# valores iniciales (una sola vez)
_init = s.PRESET_BAJO
for k, v in {"limite": _init["limite"], "edad": _init["edad"], "sexo": _init["sexo"],
             "educacion": _init["educacion"], "estado": _init["estado_civil"],
             "deuda": _init["deuda_mensual"], "pago": _init["pago_mensual"]}.items():
    st.session_state.setdefault(k, v)
for i in range(6):
    st.session_state.setdefault(f"pay{i}", _init["atrasos"][i])

b1, b2, b3, _ = st.columns([1, 1, 1, 2])
b1.button("🟢 Ejemplo: bajo riesgo", on_click=aplicar_preset, args=(s.PRESET_BAJO,))
b2.button("🟠 Ejemplo: riesgo moderado", on_click=aplicar_preset, args=(s.PRESET_MODERADO,))
b3.button("🔴 Ejemplo: alto riesgo", on_click=aplicar_preset, args=(s.PRESET_ALTO,))

col_in, col_out = st.columns([1, 1.05], gap="large")

# ================= ENTRADA =================
with col_in:
    st.markdown('<div class="eyebrow">Entrada</div>', unsafe_allow_html=True)
    st.subheader("Perfil del cliente")
    with st.container(border=True):
        limite = st.number_input("Límite de crédito (NT$)", 10000, 1000000, step=10000, key="limite")
        c1, c2 = st.columns(2)
        edad = c1.number_input("Edad", 18, 90, key="edad")
        sexo = c2.selectbox("Sexo", [1, 2], key="sexo",
                            format_func=lambda x: "Hombre" if x == 1 else "Mujer")
        c3, c4 = st.columns(2)
        educacion = c3.selectbox("Educación", [1, 2, 3, 4], key="educacion",
                                 format_func=lambda x: {1:"Posgrado",2:"Universidad",3:"Secundaria",4:"Otros"}[x])
        estado = c4.selectbox("Estado civil", [1, 2, 3], key="estado",
                              format_func=lambda x: {1:"Casado/a",2:"Soltero/a",3:"Otros"}[x])
        c5, c6 = st.columns(2)
        deuda = c5.number_input("Deuda mensual prom. (NT$)", 0, 1000000, step=1000, key="deuda")
        pago = c6.number_input("Pago mensual prom. (NT$)", 0, 1000000, step=1000, key="pago")

        st.markdown("**Estado de pago — últimos 6 meses**")
        st.caption("Selecciona el estado de cada mes (Mes 1 = el más reciente).")
        ESTADOS = {-2: "Sin uso", -1: "Al día", 0: "Mínimo",
                   1: "1 mes tarde", 2: "2 meses", 3: "3 meses", 4: "4 meses",
                   5: "5 meses", 6: "6 meses", 7: "7 meses", 8: "8 meses"}
        opciones = list(ESTADOS.keys())
        atrasos = []
        fila1 = st.columns(3)
        fila2 = st.columns(3)
        for i in range(6):
            col = fila1[i] if i < 3 else fila2[i - 3]
            atrasos.append(col.selectbox(
                f"Mes {i+1}" + (" (último)" if i == 0 else ""),
                opciones, format_func=lambda x: ESTADOS[x], key=f"pay{i}"))

# ---- predicción ----
X = s.build_features(limite, edad, sexo, educacion, estado, atrasos, deuda, pago)
prob = s.predict_proba(X)
texto, color = p.verdict(prob)

# ================= RESULTADO =================
with col_out:
    st.markdown('<div class="eyebrow">Resultado</div>', unsafe_allow_html=True)
    st.subheader("Evaluación de riesgo")
    with st.container(border=True):
        gcol, scol = st.columns([1.2, 1])
        with gcol:
            st.pyplot(p.gauge(prob), use_container_width=True)
        with scol:
            st.markdown(
                f'<div style="padding-top:.6rem"><div class="score" style="color:{color}">{prob:.0%}</div>'
                f'<div class="score-sub">probabilidad de incumplimiento</div>'
                f'<span class="pill" style="background:{color}1A;color:{color}">{texto}</span></div>',
                unsafe_allow_html=True)
        st.markdown("##### ¿Por qué? Factores que más pesaron")
        st.pyplot(p.explanation_chart(s.explain(X, 8), s.label), use_container_width=True)

st.caption("SkyScore · Random Forest sobre *Default of Credit Card Clients* (UCI) · explicabilidad con SHAP · "
           "demo de portafolio, no es una herramienta de crédito real · por Cielo Chávez · github.com/skyluw")
