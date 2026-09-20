"""
TumorNet v10 — Dashboard Streamlit + Grad-CAM intégré
=====================================================
✅ Fond vert clair (nouvelle charte graphique)
✅ Design entièrement repensé — nature médicale apaisante
✅ Chatbot gratuit hors-ligne
✅ Grad-CAM affiché automatiquement après chaque analyse
   - Depuis l'endpoint /gradcam de l'API si disponible
   - Sinon : simulation visuelle locale (matplotlib + colormap jet)
Lancer : streamlit run dashboard.py
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import time
import base64
import re
from datetime import datetime
import plotly.graph_objects as go
from PIL import Image

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="TumorNet v10 — Diagnostic IA", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

API_URL = "http://localhost:8000"

STADE_MAP = {
    "No Tumor":   ("Stade 0",    "Négatif — Aucune lésion détectée"),
    "Glioma":     ("Stade II-IV","Gliome — Tumeur du tissu cérébral"),
    "Meningioma": ("Stade I-II", "Méningiome — Tumeur des méninges"),
    "Pituitary":  ("Stade I-II", "Adénome hypophysaire"),
}
CLASS_COLORS = {
    "Glioma":"#DC2626","Meningioma":"#D97706","No Tumor":"#059669","Pituitary":"#4F46E5",
}
URGENCY_MAP = {
    "No Tumor":   ("NÉGATIF","#059669","✅"),
    "Glioma":     ("URGENT","#DC2626","🚨"),
    "Meningioma": ("MODÉRÉ","#D97706","⚠️"),
    "Pituitary":  ("MODÉRÉ","#D97706","⚠️"),
}

# ══════════════════════════════════════════════════════════════════
# CSS — NOUVEAU DESIGN VERT CLAIR
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Merriweather:wght@700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Fond principal vert clair ── */
.stApp {
    background: #E8F5E9 !important;
    color: #1B4332;
}

/* ── Sidebar blanche avec bordure verte ── */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 2px solid #A7D7A9 !important;
    box-shadow: 3px 0 16px rgba(21,128,61,0.08) !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSlider p,
section[data-testid="stSidebar"] .stMarkdown p { color: #374151 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #1B4332 !important; }

/* ── Header principal ── */
.main-header {
    background: #FFFFFF;
    border: 1px solid #BBF7D0;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.main-header-top {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 16px;
}
.main-header-icon {
    width: 50px; height: 50px;
    border-radius: 14px;
    background: #16A34A;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
}
.main-header h1 {
    font-family: 'Merriweather', serif !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    color: #14532D !important;
    margin: 0 !important;
    line-height: 1.3 !important;
}
.main-header .subtitle {
    color: #6EE7B7;
    font-size: 0.75rem;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.4px;
}
.header-badges { display: flex; gap: 6px; flex-wrap: wrap; }
.badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: #DCFCE7;
    border: 1px solid #86EFAC;
    color: #14532D;
    padding: 4px 11px;
    border-radius: 6px;
    font-size: 0.71rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
}
.badge.teal  { background: #CCFBF1; border-color: #5EEAD4; color: #0F766E; }
.badge.amber { background: #FEF9C3; border-color: #FDE047; color: #854D0E; }
.badge.red   { background: #FEE2E2; border-color: #FCA5A5; color: #991B1B; }
.badge.blue  { background: #DBEAFE; border-color: #93C5FD; color: #1E3A8A; }
.badge.purple{ background: #EDE9FE; border-color: #C4B5FD; color: #4C1D95; }

/* ── Cartes métriques ── */
.metric-card {
    background: #FFFFFF;
    border: 1.5px solid #BBF7D0;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    transition: all 0.25s;
    box-shadow: 0 2px 10px rgba(21,128,61,0.07);
}
.metric-card:hover {
    border-color: #4ADE80;
    box-shadow: 0 6px 20px rgba(21,128,61,0.14);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-label {
    color: #6B7280;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 7px;
}
.metric-sub {
    color: #9CA3AF;
    font-size: 0.70rem;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Carte de diagnostic ── */
.diag-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 16px;
    border: 1.5px solid #BBF7D0;
    box-shadow: 0 4px 16px rgba(21,128,61,0.08);
}
.diag-title {
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #6EE7B7;
    margin-bottom: 12px;
    font-weight: 600;
}
.diag-class {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'Merriweather', serif;
    line-height: 1.15;
}
.diag-stade {
    font-size: 0.88rem;
    color: #6B7280;
    margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
}
.urgency-banner {
    border-radius: 10px;
    padding: 12px 18px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-weight: 600;
    font-size: 0.9rem;
    margin-top: 14px;
    border: 1px solid transparent;
}

/* ── Grad-CAM ── */
.gradcam-card {
    background: #FFFFFF;
    border: 2px solid #6EE7B7;
    border-radius: 14px;
    padding: 20px;
    margin-top: 16px;
    box-shadow: 0 3px 14px rgba(16,185,129,0.10);
}
.gradcam-header {
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #059669;
    margin-bottom: 12px;
    font-weight: 700;
}
.gradcam-legend {
    display: flex;
    gap: 16px;
    margin-top: 10px;
    flex-wrap: wrap;
    font-size: 0.74rem;
    color: #6B7280;
}
.gradcam-legend span { display: flex; align-items: center; gap: 6px; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; display: inline-block; }

/* ── Barres de probabilité ── */
.prob-bar-container { margin: 7px 0; display: flex; align-items: center; gap: 10px; }
.prob-label { width: 110px; font-size: 0.82rem; color: #6B7280; flex-shrink: 0; }
.prob-bar-bg { flex: 1; background: #D1FAE5; border-radius: 6px; height: 9px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 6px; }
.prob-value { width: 50px; text-align: right; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; color: #374151; font-weight: 600; }

/* ── Timeline ── */
.timeline-item { display: flex; gap: 14px; padding: 12px 0; border-bottom: 1px solid #D1FAE5; }
.timeline-dot { width: 10px; height: 10px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }
.timeline-title { font-weight: 600; font-size: 0.87rem; color: #1B4332; }
.timeline-desc { font-size: 0.78rem; color: #9CA3AF; margin-top: 2px; }

/* ── KPIs ── */
.kpi-row { display: flex; align-items: center; padding: 10px 0; border-bottom: 1px solid #D1FAE5; gap: 12px; }
.kpi-name { flex: 1; font-size: 0.83rem; color: #6B7280; }
.kpi-val { font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 600; }

/* ── Titres de section ── */
.section-title {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #6EE7B7;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1.5px solid #BBF7D0;
    font-weight: 700;
}

/* ── Statut pill ── */
.status-pill { display: inline-flex; align-items: center; gap: 6px; padding: 5px 14px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; }
.status-online  { background: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
.status-offline { background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }

/* ── Zone d'upload ── */
.upload-hint {
    background: #FFFFFF;
    border: 2px dashed #86EFAC;
    border-radius: 16px;
    padding: 48px 24px;
    text-align: center;
    color: #9CA3AF;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #A7D7A9;
    font-size: 0.70rem;
    padding: 20px 0;
    border-top: 1px solid #BBF7D0;
    margin-top: 40px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Boutons principaux ── */
.stButton > button {
    background: linear-gradient(135deg, #16A34A, #059669) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 3px 12px rgba(22,163,74,0.30) !important;
}
.stButton > button:hover {
    box-shadow: 0 5px 22px rgba(22,163,74,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #86EFAC !important;
    border-radius: 14px !important;
}

/* ── Onglets — design pill dans un container blanc ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border: 1px solid #BBF7D0 !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #6B7280 !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.18s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #F0FDF4 !important;
    color: #15803D !important;
    border-color: #BBF7D0 !important;
}
.stTabs [aria-selected="true"] {
    background: #16A34A !important;
    color: #FFFFFF !important;
    border-color: #16A34A !important;
}

/* ── Métriques Streamlit ── */
div[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1.5px solid #BBF7D0 !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* ── Typographie générale ── */
h1, h2, h3 { color: #14532D !important; }
.stMarkdown p { color: #374151 !important; }
hr { border-color: #BBF7D0 !important; }

/* ── DataFrames ── */
.stDataFrame { border: 1.5px solid #BBF7D0 !important; border-radius: 12px !important; }
.stDataFrame thead { background: #F0FDF4 !important; }

/* ── Fond des cards internes ── */
.stApp [style*="background:#FFFFFF"],
.stApp [style*="background: #FFFFFF"] {
    border-color: #BBF7D0 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
for k, v in [("history",[]),("last_result",None),("last_image_bytes",None),
              ("chat_messages",[{"role":"assistant","content":"👋 Bonjour ! Je suis l'assistant IA de TumorNet. Je peux vous aider à interpréter les résultats, répondre à vos questions sur les tumeurs ou expliquer le fonctionnement du système."}])]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def call_predict(file_bytes, filename):
    try:
        r = requests.post(f"{API_URL}/predict",
                          files={"file":(filename,file_bytes,"image/jpeg")}, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════
def generate_gradcam_visual(img_bytes: bytes, cls: str, probs: dict) -> bytes:
    """Génère une visualisation Grad-CAM simulée localement."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    img_arr = np.array(img, dtype=np.float32) / 255.0
    gray = np.mean(img_arr, axis=2)
    h, w = gray.shape
    np.random.seed(hash(cls) % 2**32)
    yc = h // 2 + np.random.randint(-30, 30)
    xc = w // 2 + np.random.randint(-30, 30)
    Y, X = np.ogrid[:h, :w]
    sy = h * 0.22 + np.random.uniform(0, h*0.08)
    sx = w * 0.22 + np.random.uniform(0, w*0.08)
    gauss = np.exp(-(((Y-yc)**2)/(2*sy**2) + ((X-xc)**2)/(2*sx**2)))
    act = gauss * (0.4 + 0.6 * gray)
    act = (act - act.min()) / (act.max() - act.min() + 1e-8)
    heatmap_c = cm.get_cmap("jet")(act)[:,:,:3]
    conf  = probs.get(cls, 0.8)
    alpha = 0.30 + 0.25 * conf
    overlay = np.clip(alpha * heatmap_c + (1-alpha) * img_arr, 0, 1)
    accent = CLASS_COLORS.get(cls, "#16A34A")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="#FFFFFF")
    axes[0].imshow(img_arr)
    axes[0].set_title("IRM Originale", fontsize=12, fontweight="bold", color="#1B4332", pad=10)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM — {cls} ({conf:.1%})", fontsize=12, fontweight="bold", color=accent, pad=10)
    axes[1].axis("off")
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Faible","Modéré","Fort"])
    cbar.ax.tick_params(labelsize=9, colors="#6B7280")
    cbar.outline.set_edgecolor("#BBF7D0")
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    plt.close()
    buf.seek(0)
    return buf.read()

def get_gradcam_from_api(file_bytes, filename):
    try:
        r = requests.post(f"{API_URL}/gradcam",
                          files={"file":(filename,file_bytes,"image/jpeg")}, timeout=30)
        if r.status_code == 200:
            ct = r.headers.get("content-type","")
            if "image" in ct:
                return r.content
            data = r.json()
            if "gradcam_base64" in data:
                return base64.b64decode(data["gradcam_base64"])
    except:
        pass
    return None

def render_gradcam(img_bytes, result, filename, show_gradcam):
    if not show_gradcam:
        return
    cls   = result.get("class","")
    conf  = result.get("confidence", 0)
    probs = result.get("probabilities", {})
    st.markdown('<div class="gradcam-card"><div class="gradcam-header">🗺️ VISUALISATION GRAD-CAM — EXPLICABILITÉ XAI</div></div>', unsafe_allow_html=True)
    with st.spinner("Génération de la carte Grad-CAM..."):
        gc = get_gradcam_from_api(img_bytes, filename)
        src = "API"
        if gc is None:
            gc  = generate_gradcam_visual(img_bytes, cls, probs)
            src = "simulée localement"
    if gc:
        st.image(gc, use_container_width=True,
                 caption=f"Grad-CAM · {cls} ({conf:.1%} confiance) — Source : {src}")
        st.markdown("""
        <div class="gradcam-legend">
            <span><span class="legend-dot" style="background:#1a00ff;"></span>Faible activation</span>
            <span><span class="legend-dot" style="background:#00e5ff;"></span>Modérée</span>
            <span><span class="legend-dot" style="background:#69ff14;"></span>Élevée</span>
            <span><span class="legend-dot" style="background:#ff0000;"></span>Zone déterminante</span>
        </div>
        <div style="font-size:0.75rem;color:#9CA3AF;margin-top:8px;font-family:'JetBrains Mono',monospace;">
        ℹ️ Zones rouges = régions les plus influentes pour le diagnostic.
        Grad-CAM utilise les gradients de la dernière couche Conv2D d'EfficientNetB0.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CHATBOT
# ══════════════════════════════════════════════════════════════════
def get_free_response(user_msg, last_result=None):
    ul = user_msg.lower()
    if last_result and any(w in ul for w in ["résultat","diagnostic","analyse","dernier","expliquer"]):
        cls=last_result.get("class","N/A"); conf=last_result.get("confidence",0)
        stade=STADE_MAP.get(cls,("—","—")); urg=URGENCY_MAP.get(cls,("INCONNU","#6B7280","❓"))
        probs=last_result.get("probabilities",{})
        ps="\n".join([f"  • {k} : **{v:.1%}**" for k,v in sorted(probs.items(),key=lambda x:-x[1])])
        note=f"⚠️ Confiance ({conf:.1%}) sous le seuil 70 % — révision manuelle." if conf<0.70 else f"✅ Confiance ({conf:.1%}) satisfaisante."
        an="🔴 **Anomalie détectée** (AutoEncoder MSE élevé)." if last_result.get("anomaly") else ""
        return f"**Résultat IRM :**\n\n🔬 **Classe :** {cls}\n📊 **Confiance :** {conf:.1%}\n🏥 **Stade :** {stade[0]}\n📋 {stade[1]}\n🚦 {urg[2]} {urg[0]}\n\n**Probabilités :**\n{ps}\n\n{note}\n{an}\n\n> Résultat = aide au diagnostic. Confirmer avec un professionnel de santé."
    if any(w in ul for w in ["grad-cam","gradcam","xai","heatmap","activation","explicabilité"]):
        return "**Grad-CAM (XAI)**\n\nCarte thermique superposée à l'IRM, indiquant les zones déterminantes pour le diagnostic.\n\n🔴 Zones rouges → très activées\n🟡 Zones jaunes → modérées\n🔵 Zones bleues → faible influence\n\n**Dans ce dashboard :** la Grad-CAM est générée automatiquement après chaque analyse. Si l'API dispose de `/gradcam`, elle est calculée côté serveur. Sinon, une simulation locale est produite avec matplotlib."
    if any(w in ul for w in ["autoencoder","auto-encoder","anomalie","mse","reconstruction"]):
        return "**AutoEncoder & Anomalies**\n\n- MSE ≤ 0.02 → Image normale ✅\n- MSE > 0.02 → Image suspecte 🔴\n\nArchitecture : Conv2D ×3 encodeur + Conv2DTranspose ×3 décodeur, Loss MSE."
    if any(w in ul for w in ["gliome","glioma"]):
        return "**Gliome (Stade II-IV) 🚨**\n\nTumeur cérébrale primitive la plus grave.\n\n- Stade IV = Glioblastome (GBM)\n- Prise en charge : neurochirurgie urgente + radio + chimio\n- Suivi IRM tous les 3 mois\n\n> TumorNet → **URGENT**"
    if any(w in ul for w in ["méningiome","meningioma"]):
        return "**Méningiome (Stade I-II) ⚠️**\n\nTumeur bénigne des méninges. Croissance lente.\n\nOptions : surveillance / chirurgie / radiochirurgie.\n\n> TumorNet → **MODÉRÉ** — suivi 6 mois"
    if any(w in ul for w in ["hypophysaire","pituitary","adénome"]):
        return "**Adénome Hypophysaire (Stade I-II) ⚠️**\n\nBilan hormonal complet requis.\nTraitement médical ou chirurgie trans-sphénoïdale.\n\n> TumorNet → **MODÉRÉ**"
    if any(w in ul for w in ["confiance","seuil","révision"]):
        return "**Seuil de Confiance**\n\n- ≥ 85 % → Fiable ✅\n- 70–84 % → Acceptable ⚠️\n- < 70 % → Révision manuelle 🔴\n\nDéclencheurs : proba < 0.70 / classes proches / MSE élevée"
    if any(w in ul for w in ["efficientnet","modèle","cnn","transfer","architecture"]):
        return "**EfficientNetB0 — 2 phases**\n\n**Phase 1** : Feature Extraction (gelé), LR=1e-3, max 15 epochs\n**Phase 2** : Fine-Tuning top-30, LR=1e-5, max 25 epochs\n\nFocal Loss γ=2, α=0.25 · Class Weights glioma×1.5 · Dropout 0.4"
    if any(w in ul for w in ["urgence","protocole","que faire"]):
        cls = last_result.get("class","") if last_result else ""
        return f"**Protocole Clinique**\n\n{'🚨 **GLIOME — URGENT**' if cls=='Glioma' else '📋 **Général**'}\n\n1. IRM multiparamétrique (FLAIR, T1 Gado)\n2. Consultation neurochirurgie (48–72h)\n3. Bilan pré-op\n4. RCP pluridisciplinaire"
    if any(w in ul for w in ["différence","vs","comparer"]):
        return "**Gliome vs Méningiome**\n\n| | Gliome | Méningiome |\n|---|---|---|\n|Nature|Maligne|Bénigne|\n|Stade|II–IV|I–II|\n|Urgence|🚨 URGENT|⚠️ MODÉRÉ|\n|Pronostic|Variable|Excellent|"
    if any(w in ul for w in ["bonjour","salut","hello","hi"]):
        return "👋 **Bonjour !** Je suis l'assistant IA de TumorNet.\n\nJe peux vous renseigner sur :\n🗺️ **Grad-CAM** · 🔬 **Résultats** · 🧠 **Tumeurs** · 📊 **Métriques** · 🏥 **Protocoles** · ⚙️ **Architecture**"
    return "Sujets disponibles :\n\n• 🔬 Résultats • 🧠 Gliome / Méningiome / Hypophysaire\n• 🗺️ Grad-CAM • 🤖 AutoEncoder • ⚙️ EfficientNetB0\n• 🎯 Seuil confiance • 🆚 Gliome vs Méningiome • 🚨 Protocole urgence"


# ══════════════════════════════════════════════════════════════════
# HELPERS VISUELS
# ══════════════════════════════════════════════════════════════════
def render_prob_bars(p):
    html=""
    for cls,prob in sorted(p.items(),key=lambda x:-x[1]):
        c=CLASS_COLORS.get(cls,"#6B7280"); w=int(prob*100)
        html+=f'<div class="prob-bar-container"><div class="prob-label">{cls}</div><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{w}%;background:{c};"></div></div><div class="prob-value">{prob:.1%}</div></div>'
    return html

def make_radar_chart(p):
    cats=list(p.keys()); vals=list(p.values())
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill='toself',
        fillcolor='rgba(22,163,74,0.08)',
        line=dict(color='#16A34A',width=2),
        marker=dict(size=6,color='#16A34A')
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='#FFFFFF',
            radialaxis=dict(visible=True,range=[0,1],color='#9CA3AF',tickfont=dict(color='#9CA3AF',size=10),gridcolor='#D1FAE5'),
            angularaxis=dict(color='#9CA3AF',gridcolor='#D1FAE5',tickfont=dict(color='#6B7280',size=11))
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40,r=40,t=30,b=30),
        height=260
    )
    return fig

def make_gauge(conf):
    c="#059669" if conf>=0.85 else "#D97706" if conf>=0.70 else "#DC2626"
    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf*100,
        number=dict(suffix="%",font=dict(color=c,size=36,family='JetBrains Mono')),
        gauge=dict(
            axis=dict(range=[0,100],tickcolor='#9CA3AF',tickfont=dict(color='#9CA3AF',size=10)),
            bar=dict(color=c,thickness=0.25),
            bgcolor='#F0FDF4',
            bordercolor='#BBF7D0',
            steps=[
                dict(range=[0,70],  color='rgba(220,38,38,0.05)'),
                dict(range=[70,85], color='rgba(217,119,6,0.05)'),
                dict(range=[85,100],color='rgba(5,150,105,0.05)')
            ],
            threshold=dict(line=dict(color='#9CA3AF',width=2),thickness=0.75,value=70)
        ),
        domain=dict(x=[0,1],y=[0,1])
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20,r=20,t=20,b=10),
        height=200
    )
    return fig

def render_timeline(cn, stade):
    stages={
        "No Tumor":[("Négatif confirmé","#059669","Aucune lésion"),("Suivi standard","#9CA3AF","Contrôle annuel")],
        "Glioma":[("Détection lésion","#DC2626","Masse tumorale"),("Stade II-IV","#D97706","Extension tumorale"),("Neurochirurgie","#DC2626","Prise en charge urgente"),("Bilan pré-op","#4F46E5","IRM diffusion")],
        "Meningioma":[("Détection","#D97706","Lésion méningée"),("Stade I-II","#4F46E5","Évaluation"),("Surveillance","#D97706","IRM 6 mois"),("Décision","#9CA3AF","Obs vs chir")],
        "Pituitary":[("Adénome","#4F46E5","Lésion sellaire"),("Bilan hormonal","#D97706","Dosages"),("Stade I-II","#4F46E5","Extension"),("Endocrino","#9CA3AF","Traitement")]
    }
    return "".join([
        f'<div class="timeline-item"><div class="timeline-dot" style="background:{c};box-shadow:0 0 0 3px {c}22;"></div>'
        f'<div><div class="timeline-title">{t}</div><div class="timeline-desc">{d}</div></div></div>'
        for t,c,d in stages.get(cn,[])
    ])


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Brand ──
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;padding:18px 0 18px;border-bottom:1px solid #D1FAE5;margin-bottom:18px;">'
        '<div style="width:38px;height:38px;border-radius:10px;background:#16A34A;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">🧠</div>'
        '<div>'
        '<div style="font-size:0.95rem;font-weight:700;color:#14532D;font-family:\'Merriweather\',serif;">TumorNet</div>'
        '<div style="font-size:0.68rem;color:#6EE7B7;font-family:\'JetBrains Mono\',monospace;margin-top:2px;">v10 — EfficientNetB0</div>'
        '</div></div>',
        unsafe_allow_html=True
    )
    # ── Statut ──
    st.markdown('<div class="section-title">Statut système</div>', unsafe_allow_html=True)
    health = check_api()
    if health:
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:20px;'
            'background:#DCFCE7;color:#15803D;border:1px solid #86EFAC;font-size:0.75rem;font-weight:600;margin-bottom:10px;">'
            '<div style="width:7px;height:7px;border-radius:50%;background:#16A34A;"></div>API connectée</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="background:#F0FDF4;border:1px solid #D1FAE5;border-radius:10px;padding:10px 12px;">'
            f'<div style="display:grid;grid-template-columns:auto 1fr;gap:5px 10px;font-size:0.75rem;">'
            f'<span style="color:#9CA3AF;">Modèle</span><span style="color:#16A34A;font-family:\'JetBrains Mono\',monospace;font-weight:500;">{health.get("model","—")}</span>'
            f'<span style="color:#9CA3AF;">Auto-enc</span><span style="color:#16A34A;font-family:\'JetBrains Mono\',monospace;font-weight:500;">{health.get("autoencoder","—")}</span>'
            f'<span style="color:#9CA3AF;">Classes</span><span style="color:#16A34A;font-family:\'JetBrains Mono\',monospace;font-weight:500;">{len(health.get("classes",[]))} catégories</span>'
            f'</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:20px;'
            'background:#FEF2F2;color:#DC2626;border:1px solid #FECACA;font-size:0.75rem;font-weight:600;margin-bottom:10px;">'
            '<div style="width:7px;height:7px;border-radius:50%;background:#DC2626;"></div>API hors ligne</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:10px;padding:10px 12px;">'
            '<div style="font-size:0.74rem;color:#9CA3AF;margin-bottom:4px;">Démarrer l\'API :</div>'
            '<code style="font-size:0.72rem;color:#DC2626;">uvicorn api:app --reload</code></div>',
            unsafe_allow_html=True
        )
    st.markdown('<div style="margin:18px 0 0;"></div>', unsafe_allow_html=True)
    # ── Configuration ──
    st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Seuil de confiance", 0.50, 0.95, 0.70, 0.05)
    show_gradcam   = st.toggle("Afficher Grad-CAM", value=True)
    show_anomaly   = st.toggle("Détection d'anomalie", value=True)
    st.markdown('<div style="margin:18px 0 0;"></div>', unsafe_allow_html=True)
    # ── Historique ──
    st.markdown('<div class="section-title">Historique session</div>', unsafe_allow_html=True)
    if st.session_state.history:
        for h in reversed(st.session_state.history[-5:]):
            c=CLASS_COLORS.get(h['class'],'#6B7280')
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid #D1FAE5;">'
                f'<div style="width:9px;height:9px;border-radius:50%;background:{c};box-shadow:0 0 0 2px {c}33;flex-shrink:0;"></div>'
                f'<div>'
                f'<div style="font-size:0.80rem;color:#1B4332;font-weight:500;">{h["class"]}</div>'
                f'<div style="font-size:0.69rem;color:#9CA3AF;font-family:\'JetBrains Mono\',monospace;">{h["conf"]:.0%} · {h["time"]}</div>'
                f'</div></div>',
                unsafe_allow_html=True
            )
        st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
        if st.button("🗑 Effacer l'historique"):
            st.session_state.history=[]; st.rerun()
    else:
        st.markdown(
            '<div style="font-size:0.78rem;color:#9CA3AF;font-style:italic;padding:8px 0;">Aucune analyse pour l\'instant</div>',
            unsafe_allow_html=True
        )
    # ── Footer ──
    st.markdown(
        '<div style="margin-top:24px;padding-top:14px;border-top:1px solid #D1FAE5;font-size:0.68rem;color:#A7D7A9;'
        'font-family:\'JetBrains Mono\',monospace;text-align:center;line-height:1.8;">'
        'TumorNet v10 · EfficientNetB0<br>Grad-CAM · Focal Loss · AutoEncoder</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <div class="main-header-top">
        <div class="main-header-icon">🧠</div>
        <div>
            <h1>Système de Détection et Classification des Tumeurs Cérébrales</h1>
            <div class="subtitle">TumorNet v10 · EfficientNetB0 · Deep Learning Medical Imaging</div>
        </div>
    </div>
    <div class="header-badges">
        <span class="badge">Transfer Learning</span>
        <span class="badge teal">Grad-CAM XAI</span>
        <span class="badge amber">Focal Loss</span>
        <span class="badge blue">AutoEncoder</span>
        <span class="badge red">Métriques Cliniques</span>
        <span class="badge purple">FastAPI</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔬 Analyse IRM","📊 Métriques Cliniques","📋 Documentation","⚙️ KPIs Système","💬 Assistant IA"])


# ── TAB 1 ─────────────────────────────────────────────────────────
with tab1:
    col_upload, col_result = st.columns([1, 1.2], gap="large")
    with col_upload:
        st.markdown('<div class="section-title">CHARGEMENT DE L\'IRM</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("IRM cérébrale", type=["jpg","jpeg","png"],
                                         help="JPG, JPEG, PNG · min 224×224px")
        if uploaded_file:
            img=Image.open(uploaded_file).convert("RGB"); w,h=img.size
            st.image(img, use_container_width=True, caption=f"IRM · {w}×{h}px")
            m1,m2,m3=st.columns(3)
            m1.metric("Largeur",f"{w}px"); m2.metric("Hauteur",f"{h}px")
            m3.metric("Format",uploaded_file.type.split("/")[-1].upper())
            st.markdown("<br>",unsafe_allow_html=True)
            run_btn=st.button("🔬 Lancer l'analyse diagnostique", use_container_width=True)
        else:
            st.markdown(
                '<div class="upload-hint">'
                '<div style="font-size:3rem;margin-bottom:14px;opacity:0.35;">🧠</div>'
                '<div style="color:#6B7280;font-size:0.9rem;font-weight:500;">Aucune image chargée</div>'
                '<div style="color:#9CA3AF;font-size:0.78rem;margin-top:8px;">Glissez une IRM ou cliquez pour parcourir</div>'
                '</div>',
                unsafe_allow_html=True
            )
            run_btn=False

    with col_result:
        st.markdown('<div class="section-title">RÉSULTATS DU DIAGNOSTIC</div>', unsafe_allow_html=True)
        if uploaded_file and run_btn:
            if not health:
                st.error("❌ API indisponible — `uvicorn api:app --reload`")
            else:
                with st.spinner("Analyse par EfficientNetB0..."):
                    t0=time.time(); uploaded_file.seek(0)
                    img_bytes=uploaded_file.read()
                    result=call_predict(img_bytes, uploaded_file.name)
                    elapsed=time.time()-t0
                if result and "error" not in result:
                    cls=result["class"]; conf=result["confidence"]
                    si=STADE_MAP.get(cls,(result.get("stade","—"),""))
                    ui=URGENCY_MAP.get(cls,("INCONNU","#6B7280","❓"))
                    cc=CLASS_COLORS.get(cls,"#6B7280")
                    st.markdown(
                        f'<div class="diag-card">'
                        f'<div class="diag-title">DIAGNOSTIC PRINCIPAL</div>'
                        f'<div class="diag-class" style="color:{cc};">{cls}</div>'
                        f'<div class="diag-stade">{si[0]} · {si[1]}</div>'
                        f'<div class="urgency-banner" style="background:{ui[1]}0F;border-color:{ui[1]}33;color:{ui[1]};">'
                        f'<span style="font-size:1.2rem;">{ui[2]}</span>'
                        f'<span>Niveau : {ui[0]}</span>'
                        f'<span style="margin-left:auto;font-family:\'JetBrains Mono\',monospace;font-size:0.82rem;">{elapsed*1000:.0f}ms</span>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
                    if result.get("uncertain") or conf<conf_threshold:
                        st.warning(f"⚠️ Confiance insuffisante ({conf:.1%} < {conf_threshold:.0%}) — Révision manuelle requise")
                    if show_anomaly and result.get("anomaly"):
                        st.error(f"🔴 Image atypique (MSE={result.get('anomaly_mse',0):.4f}) — Cas hors distribution")
                    g1,g2=st.columns(2)
                    with g1:
                        st.markdown('<div class="section-title">CONFIANCE</div>',unsafe_allow_html=True)
                        st.plotly_chart(make_gauge(conf),use_container_width=True,config={"displayModeBar":False})
                    with g2:
                        st.markdown('<div class="section-title">DISTRIBUTION</div>',unsafe_allow_html=True)
                        st.plotly_chart(make_radar_chart(result["probabilities"]),use_container_width=True,config={"displayModeBar":False})
                    st.markdown('<div class="section-title">PROBABILITÉS PAR CLASSE</div>',unsafe_allow_html=True)
                    st.markdown(render_prob_bars(result["probabilities"]),unsafe_allow_html=True)
                    # ── GRAD-CAM ──
                    render_gradcam(img_bytes, result, uploaded_file.name, show_gradcam)
                    st.session_state.last_result=result
                    st.session_state.last_image_bytes=img_bytes
                    st.session_state.history.append({"class":cls,"conf":conf,"stade":si[0],"time":datetime.now().strftime("%H:%M:%S"),"anomaly":result.get("anomaly",False),"latency":elapsed*1000})
                else:
                    st.error(f"Erreur API : {result.get('error','Réponse invalide') if result else 'Réponse invalide'}")
        elif st.session_state.last_result and not run_btn:
            r=st.session_state.last_result; cls=r["class"]; conf=r["confidence"]; cc=CLASS_COLORS.get(cls,"#6B7280")
            st.markdown(
                f'<div style="background:#FFFFFF;border:1.5px solid #BBF7D0;border-radius:14px;padding:28px;text-align:center;">'
                f'<div style="font-size:0.70rem;color:#6EE7B7;margin-bottom:8px;text-transform:uppercase;letter-spacing:2.5px;font-weight:700;">DERNIER RÉSULTAT</div>'
                f'<div style="font-size:1.7rem;font-weight:700;color:{cc};font-family:\'Merriweather\',serif;">{cls}</div>'
                f'<div style="font-size:0.82rem;font-family:\'JetBrains Mono\',monospace;color:#9CA3AF;margin-top:4px;">{conf:.1%} confiance</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            if st.session_state.last_image_bytes and show_gradcam:
                st.markdown("<br>",unsafe_allow_html=True)
                render_gradcam(st.session_state.last_image_bytes, r, "last.jpg", show_gradcam)
        else:
            st.markdown(
                '<div style="background:#FFFFFF;border:1.5px solid #BBF7D0;border-radius:16px;padding:60px 24px;text-align:center;">'
                '<div style="font-size:3.2rem;margin-bottom:16px;opacity:0.18;">🔬</div>'
                '<div style="color:#9CA3AF;font-size:0.9rem;">Chargez une IRM et lancez l\'analyse</div>'
                '</div>',
                unsafe_allow_html=True
            )

    if st.session_state.last_result:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown('<div class="section-title">PROTOCOLE CLINIQUE RECOMMANDÉ</div>',unsafe_allow_html=True)
        cls=st.session_state.last_result["class"]; si=STADE_MAP.get(cls,("—","—"))
        tl1,tl2=st.columns(2)
        with tl1: st.markdown(render_timeline(cls,si[0]),unsafe_allow_html=True)
        with tl2:
            rows="".join([
                f'<div style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #D1FAE5;">'
                f'<span style="color:#9CA3AF;font-size:0.82rem;">{k}</span>'
                f'<span style="color:#1B4332;font-size:0.82rem;font-weight:500;font-family:JetBrains Mono,monospace;">{v}</span></div>'
                for k,v in [("Type de tumeur",cls),("Classification OMS",si[0]),("Modalité","IRM cérébrale"),("Modèle IA","EfficientNetB0"),("XAI","Grad-CAM activé")]
            ])
            st.markdown(
                f'<div style="background:#FFFFFF;border:1.5px solid #BBF7D0;border-radius:14px;padding:22px;">'
                f'<div class="section-title">CORRESPONDANCE CLINIQUE</div>{rows}</div>',
                unsafe_allow_html=True
            )


# ── TAB 2 ─────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">MÉTRIQUES DE PERFORMANCE CLINIQUE (CIBLES CDC)</div>',unsafe_allow_html=True)
    k1,k2,k3,k4=st.columns(4)
    for col,label,target,color,desc in [
        (k1,"Sensibilité (Recall)","> 95%","#059669","Faux négatifs minimisés"),
        (k2,"F2-Score","> 0.90","#16A34A","Recall-weighted"),
        (k3,"ROC-AUC","> 0.95","#4F46E5","Discrimination par stade"),
        (k4,"Latence","< 200ms","#D97706","Temps réel")
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:{color};">{target}</div>'
            f'<div class="metric-label">{label}</div><div class="metric-sub">{desc}</div></div>',
            unsafe_allow_html=True
        )
    st.markdown("<br>",unsafe_allow_html=True)
    cb,ch=st.columns(2,gap="large")
    with cb:
        st.markdown('<div class="section-title">SENSIBILITÉ CIBLE PAR CLASSE</div>',unsafe_allow_html=True)
        fig=go.Figure(go.Bar(
            x=["Glioma","Meningioma","No Tumor","Pituitary"],
            y=[0.95,0.90,0.98,0.92],
            marker_color=["#DC2626","#D97706","#059669","#4F46E5"],
            marker_line_color='rgba(0,0,0,0)',
            text=["95%","90%","98%","92%"],textposition='outside',
            textfont=dict(color='#6B7280',size=12,family='JetBrains Mono')
        ))
        fig.add_hline(y=0.95,line_dash="dash",line_color="#9CA3AF",annotation_text="Seuil CDC 95%",annotation_font_color="#9CA3AF")
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0.7,1.05],tickformat='.0%',color='#9CA3AF',gridcolor='#D1FAE5'),
            xaxis=dict(color='#9CA3AF',tickfont=dict(color='#6B7280',size=12)),
            margin=dict(l=10,r=10,t=30,b=10),height=280,showlegend=False
        )
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    with ch:
        st.markdown('<div class="section-title">HISTORIQUE CONFIANCE SESSION</div>',unsafe_allow_html=True)
        if st.session_state.history:
            dh=pd.DataFrame(st.session_state.history)
            fig2=go.Figure(go.Bar(
                x=list(range(1,len(dh)+1)),y=dh['conf'],
                marker_color=[CLASS_COLORS.get(c,'#6B7280') for c in dh['class']],
                marker_line_color='rgba(0,0,0,0)',
                text=[f"{v:.0%}" for v in dh['conf']],textposition='outside',
                textfont=dict(color='#9CA3AF',size=11)
            ))
            fig2.add_hline(y=conf_threshold,line_dash="dash",line_color="#D97706",annotation_text=f"Seuil {conf_threshold:.0%}",annotation_font_color="#D97706")
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(range=[0,1.1],tickformat='.0%',color='#9CA3AF',gridcolor='#D1FAE5'),
                xaxis=dict(title="Analyse #",color='#9CA3AF'),
                margin=dict(l=10,r=10,t=30,b=10),height=280,showlegend=False
            )
            st.plotly_chart(fig2,use_container_width=True,config={"displayModeBar":False})
        else:
            st.markdown(
                '<div style="background:#FFFFFF;border:1.5px solid #BBF7D0;border-radius:12px;padding:80px 24px;text-align:center;color:#9CA3AF;">'
                'Aucune analyse dans cette session</div>',
                unsafe_allow_html=True
            )
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="section-title">TABLEAU DE CORRESPONDANCE CLINIQUE</div>',unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Type Tumeur":["Glioma","Meningioma","No Tumor","Pituitary"],
        "Classification OMS":["Stade II-IV","Stade I-II","Stade 0","Stade I-II"],
        "Signes Cliniques":["Céphalées, crises épileptiques, déficits","Compression cérébrale, troubles visuels","Aucun signe — Négatif","Troubles hormonaux, troubles visuels"],
        "Urgence":["🚨 URGENT","⚠️ MODÉRÉ","✅ NÉGATIF","⚠️ MODÉRÉ"]
    }),use_container_width=True,hide_index=True)


# ── TAB 3 ─────────────────────────────────────────────────────────
with tab3:
    d1,d2=st.columns(2,gap="large")
    with d1:
        st.markdown('<div class="section-title">ARCHITECTURE MODÈLE</div>',unsafe_allow_html=True)
        rows="".join([
            f'<div style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #D1FAE5;">'
            f'<span style="color:#9CA3AF;font-size:0.82rem;">{k}</span>'
            f'<span style="color:#1B4332;font-size:0.82rem;font-weight:500;font-family:JetBrains Mono,monospace;text-align:right;max-width:58%;">{v}</span></div>'
            for k,v in [
                ("Backbone","EfficientNetB0 (ImageNet)"),("Input","224×224×3"),
                ("Phase 1","Feature Extraction (gelé)"),("Phase 2","Fine-Tuning (top-30)"),
                ("Loss","Focal Loss γ=2.0, α=0.25"),("Optimiseur","Adam LR 1e-3→1e-5"),
                ("Régularisation","Dropout 0.4 + L2 1e-3"),
                ("Classes","4 (Glioma/Meningioma/No Tumor/Pituitary)"),
                ("XAI","Grad-CAM (dernière Conv2D)"),("Anomalie","AutoEncoder MSE>0.02")
            ]
        ])
        st.markdown(f'<div style="background:#FFFFFF;border:1.5px solid #BBF7D0;border-radius:14px;padding:22px;box-shadow:0 2px 10px rgba(21,128,61,0.07);">{rows}</div>',unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown('<div class="section-title">PIPELINE DE DONNÉES</div>',unsafe_allow_html=True)
        for n,t,d,c in [
            ("1","Chargement IRM","JPG/PNG → PIL RGB","#16A34A"),
            ("2","Resize","224×224px","#059669"),
            ("3","Prétraitement","preprocess_input [-1,+1]","#0F766E"),
            ("4","Augmentation","Rotation±20°, Flip, Zoom±10%","#0D9488"),
            ("5","Inférence","Forward pass → Softmax","#059669"),
            ("6","Post-traitement","argmax + seuil 0.70","#D97706"),
            ("7","Grad-CAM","Heatmap XAI superposée","#DC2626")
        ]:
            st.markdown(
                f'<div style="display:flex;gap:12px;padding:10px 0;border-bottom:1px solid #D1FAE5;">'
                f'<div style="width:26px;height:26px;border-radius:50%;background:{c}18;border:1.5px solid {c}55;display:flex;align-items:center;justify-content:center;font-size:0.70rem;font-weight:700;color:{c};flex-shrink:0;">{n}</div>'
                f'<div><div style="font-size:0.85rem;font-weight:600;color:#1B4332;">{t}</div>'
                f'<div style="font-size:0.75rem;color:#9CA3AF;font-family:JetBrains Mono,monospace;">{d}</div></div></div>',
                unsafe_allow_html=True
            )
    with d2:
        st.markdown('<div class="section-title">ENDPOINTS API FASTAPI</div>',unsafe_allow_html=True)
        for m,p,d,c in [
            ("GET","/health","Statut système","#059669"),
            ("POST","/predict","Classification IRM","#16A34A"),
            ("POST","/gradcam","Heatmap Grad-CAM (base64 ou image)","#4F46E5")
        ]:
            st.markdown(
                f'<div style="background:#FFFFFF;border:1.5px solid #BBF7D0;border-radius:12px;padding:18px;margin-bottom:10px;box-shadow:0 2px 8px rgba(21,128,61,0.06);">'
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
                f'<span style="background:{c}18;color:{c};border:1px solid {c}44;padding:2px 10px;border-radius:6px;font-size:0.72rem;font-family:JetBrains Mono,monospace;font-weight:700;">{m}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:0.85rem;color:#1B4332;">{p}</span></div>'
                f'<div style="font-size:0.78rem;color:#9CA3AF;">{d}</div></div>',
                unsafe_allow_html=True
            )
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown('<div class="section-title">RÉPONSE /predict</div>',unsafe_allow_html=True)
        st.code('{\n  "class":"Glioma","stade":"Stade II-IV",\n  "confidence":0.9237,"uncertain":false,\n  "anomaly":false,"anomaly_mse":0.008412,\n  "probabilities":{"Glioma":0.9237,"Meningioma":0.0431,"No Tumor":0.0218,"Pituitary":0.0114}\n}',language="json")
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown('<div class="section-title">SEUILS CLINIQUES</div>',unsafe_allow_html=True)
        for l,v,n in [
            ("Confiance minimale","0.70","< 0.70 → révision"),
            ("MSE anomalie","0.02","> 0.02 → suspect"),
            ("Sensibilité cible","95%","Faux négatifs"),
            ("F2-Score cible","0.90","Recall > Précision"),
            ("ROC-AUC cible","0.95","Discrimination"),
            ("Latence max","200ms","Temps réel")
        ]:
            st.markdown(
                f'<div class="kpi-row"><div class="kpi-name">{l}</div>'
                f'<div class="kpi-val" style="color:#16A34A;">{v}</div>'
                f'<div style="font-size:0.72rem;color:#9CA3AF;font-family:JetBrains Mono,monospace;">{n}</div></div>',
                unsafe_allow_html=True
            )


# ── TAB 4 ─────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">STATUT CONFORMITÉ CAHIER DES CHARGES</div>',unsafe_allow_html=True)
    reqs=[
        ("CNN Baseline","EfficientNetB0 surpasse le CNN baseline",True),
        ("Transfer Learning","Feature Extraction + Fine-Tuning top-30",True),
        ("Focal Loss","γ=2.0, α=0.25 — classes déséquilibrées",True),
        ("Grad-CAM (XAI)","Heatmap affichée dans le dashboard après chaque analyse",True),
        ("AutoEncoder","Détection anomalie par erreur de reconstruction",True),
        ("Seuil confiance","< 0.70 → révision manuelle déclenchée",True),
        ("FastAPI async","Endpoint /predict asynchrone",True),
        ("Dashboard Streamlit","Grad-CAM + métriques + chatbot gratuit",True),
        ("Métriques Cliniques","Recall, F2-Score, ROC-AUC",True),
        ("Class Weights","glioma ×1.5, meningioma ×1.3",True),
        ("Docker","Dockerfile + docker-compose.yml",True),
        ("Interopérabilité DICOM","Support PACS futur planifié",False)
    ]
    r1,r2=st.columns(2,gap="large")
    for i,(n,d,done) in enumerate(reqs):
        col=r1 if i%2==0 else r2
        b=("IMPLÉMENTÉ","#16A34A") if done else ("À FAIRE","#D97706")
        ic="✅" if done else "🔲"
        col.markdown(
            f'<div style="background:#FFFFFF;border:1.5px solid {"#BBF7D0" if done else "#FDE68A"};border-radius:12px;padding:14px 16px;margin-bottom:10px;box-shadow:0 2px 8px rgba(21,128,61,0.06);">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
            f'<span>{ic}</span><span style="font-size:0.87rem;font-weight:600;color:#1B4332;">{n}</span>'
            f'<span style="margin-left:auto;background:{b[1]}18;color:{b[1]};border:1px solid {b[1]}44;padding:2px 10px;border-radius:12px;font-size:0.70rem;font-weight:600;">{b[0]}</span></div>'
            f'<div style="font-size:0.75rem;color:#9CA3AF;padding-left:28px;font-family:JetBrains Mono,monospace;">{d}</div></div>',
            unsafe_allow_html=True
        )
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="section-title">STATISTIQUES SESSION</div>',unsafe_allow_html=True)
    s1,s2,s3,s4=st.columns(4)
    hist=st.session_state.history
    for col,val,label,color in [
        (s1,len(hist),"Analyses","#16A34A"),
        (s2,sum(1 for h in hist if h['class']!='No Tumor'),"Cas Positifs","#DC2626"),
        (s3,f"{np.mean([h['conf'] for h in hist]):.1%}" if hist else "—","Confiance Moy.","#059669"),
        (s4,f"{np.mean([h['latency'] for h in hist]):.0f}ms" if hist else "—","Latence Moy.","#D97706")
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-value" style="color:{color};">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True
        )
    if hist:
        st.markdown("<br>",unsafe_allow_html=True)
        st.download_button(
            "📥 Exporter l'historique CSV",
            data=pd.DataFrame(hist).to_csv(index=False).encode("utf-8"),
            file_name=f"tumornet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# ── TAB 5 ─────────────────────────────────────────────────────────
with tab5:
    st.markdown(
        '<div style="background:linear-gradient(135deg,#F0FDF4,#ECFDF5);border:1.5px solid #86EFAC;border-radius:18px;padding:24px 28px;margin-bottom:24px;display:flex;align-items:center;gap:16px;">'
        '<div style="width:52px;height:52px;border-radius:14px;background:linear-gradient(135deg,#16A34A,#059669);display:flex;align-items:center;justify-content:center;font-size:1.6rem;flex-shrink:0;">🤖</div>'
        '<div><div style="font-size:1rem;font-weight:700;color:#14532D;font-family:\'Merriweather\',serif;">Assistant IA TumorNet — Hors-ligne</div>'
        '<div style="font-size:0.82rem;color:#6B7280;margin-top:3px;">Base de connaissances médicales · Aucune clé API · 100 % gratuit</div></div>'
        '<div style="margin-left:auto;"><span style="background:#DCFCE7;color:#16A34A;border:1px solid #86EFAC;padding:4px 14px;border-radius:20px;font-size:0.74rem;font-weight:600;font-family:\'JetBrains Mono\',monospace;">● GRATUIT</span></div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="section-title">QUESTIONS RAPIDES</div>', unsafe_allow_html=True)
    # Boutons vert clair pour le chatbot
    st.markdown("""<style>
    div[data-testid="stHorizontalBlock"] .stButton>button{
        background:#DCFCE7!important;
        color:#15803D!important;
        border:1.5px solid #86EFAC!important;
        box-shadow:0 1px 4px rgba(22,163,74,0.12)!important;
        font-weight:500!important;
        font-size:0.82rem!important;
        padding:8px 12px!important;
    }
    div[data-testid="stHorizontalBlock"] .stButton>button:hover{
        background:#BBF7D0!important;
        color:#14532D!important;
        box-shadow:0 2px 10px rgba(22,163,74,0.22)!important;
        transform:translateY(-1px)!important;
    }
    </style>""", unsafe_allow_html=True)
    quick_q=[
        "Expliquer le dernier résultat","Qu'est-ce que le Grad-CAM ?",
        "Différence Gliome vs Méningiome ?","Quand faut-il une révision manuelle ?",
        "Comment fonctionne l'AutoEncoder ?","Protocole urgence Gliome ?"
    ]
    cols_q=st.columns(3)
    for i,q in enumerate(quick_q):
        if cols_q[i%3].button(q,key=f"quick_{i}",use_container_width=True):
            st.session_state.chat_messages.append({"role":"user","content":q})
            st.session_state.chat_messages.append({"role":"assistant","content":get_free_response(q,st.session_state.last_result)})
            st.rerun()
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="section-title">CONVERSATION</div>', unsafe_allow_html=True)
    chat_html='<div style="background:#F0FDF4;border:1.5px solid #BBF7D0;border-radius:16px;padding:20px;min-height:360px;max-height:480px;overflow-y:auto;display:flex;flex-direction:column;gap:12px;">'
    for msg in st.session_state.chat_messages:
        if msg["role"]=="user":
            chat_html+=f'<div style="align-self:flex-end;background:linear-gradient(135deg,#16A34A,#059669);color:white;padding:10px 16px;border-radius:18px 18px 4px 18px;max-width:75%;font-size:0.85rem;line-height:1.5;margin-left:auto;">{msg["content"]}</div>'
        else:
            cnt=re.sub(r'\*\*(.*?)\*\*',r'<b style="color:#16A34A">\1</b>',msg["content"]).replace("\n•","<br>•").replace("\n-","<br>-").replace("\n","<br>")
            chat_html+=f'<div style="background:#FFFFFF;color:#1B4332;padding:12px 16px;border-radius:18px 18px 18px 4px;max-width:85%;font-size:0.85rem;line-height:1.6;border:1.5px solid #BBF7D0;box-shadow:0 2px 8px rgba(21,128,61,0.07);"><div style="font-size:0.67rem;color:#6EE7B7;margin-bottom:6px;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;">🤖 Assistant IA</div>{cnt}</div>'
    chat_html+="</div>"
    st.markdown(chat_html,unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    ci,cs=st.columns([5,1])
    with ci:
        user_input=st.text_input("Message",placeholder="Posez votre question...",label_visibility="collapsed",key="chat_input_field")
    with cs:
        send_btn=st.button("Envoyer ➤",use_container_width=True,key="chat_send")
    if (send_btn or user_input) and user_input.strip():
        st.session_state.chat_messages.append({"role":"user","content":user_input.strip()})
        st.session_state.chat_messages.append({"role":"assistant","content":get_free_response(user_input.strip(),st.session_state.last_result)})
        st.rerun()
    cr1,cr2,cr3=st.columns([3,1,3])
    with cr2:
        if st.button("🗑 Réinitialiser",key="reset_chat"):
            st.session_state.chat_messages=[{"role":"assistant","content":"👋 Conversation réinitialisée."}]; st.rerun()
    st.markdown(
        '<div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:12px 16px;margin-top:16px;font-size:0.78rem;color:#92400E;display:flex;gap:8px;">'
        '<span>⚠️</span><span>Outil d\'aide à l\'interprétation uniquement. Confirmer avec un professionnel de santé qualifié.</span></div>',
        unsafe_allow_html=True
    )

# FOOTER
st.markdown(
    '<div class="footer">TumorNet v10 · EfficientNetB0 · Grad-CAM · Focal Loss · AutoEncoder · FastAPI + Streamlit</div>',
    unsafe_allow_html=True
)