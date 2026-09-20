"""
TumorNet v10 — VERSION FINALE WINDOWS LOCAL
============================================
✅ EfficientNetB0 + preprocess_input correct
✅ Grad-CAM (XAI)
✅ Auto-encodeur
✅ FastAPI (API REST)
✅ Dashboard Streamlit
✅ Métriques cliniques complètes (Recall, F2, ROC-AUC)
✅ Focal Loss
✅ Incertitude / seuil de confiance
✅ 100% compatible Windows local (pas besoin de Colab)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")  # ← important sur Windows (pas de GUI)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import classification_report, roc_auc_score, fbeta_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION GLOBALE
# ══════════════════════════════════════════════════════════════════
CONFIG = {
    "IMG_SIZE":        (224, 224),
    "BATCH_SIZE":      32,
    "EPOCHS_FEATURE":  15,
    "EPOCHS_FINETUNE": 25,
    "LR_FEATURE":      1e-3,
    "LR_FINETUNE":     1e-5,
    "NUM_CLASSES":     4,
    "DROPOUT":         0.4,
    "L2_REG":          1e-3,
    "CONF_THRESHOLD":  0.70,
    "CLASS_NAMES":     ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
    "STADE_MAP": {
        "No Tumor":   "Stade 0",
        "Glioma":     "Stade II-IV",
        "Meningioma": "Stade I-II",
        "Pituitary":  "Stade I-II",
    },
    # ── Chemins Windows ──
    "BASE_DIR":   r"C:\Users\ACER\Downloads\archive (1)",
    "MODELS_DIR": r"C:\Users\ACER\Downloads\deepfinal\models",
}

os.makedirs(CONFIG["MODELS_DIR"], exist_ok=True)
TRAIN_DIR  = os.path.join(CONFIG["BASE_DIR"], "Training")
TEST_DIR   = os.path.join(CONFIG["BASE_DIR"], "Testing")
MODEL_PATH = os.path.join(CONFIG["MODELS_DIR"], "tumor_model.h5")
AE_PATH    = os.path.join(CONFIG["MODELS_DIR"], "autoencoder.h5")


# ══════════════════════════════════════════════════════════════════
# 1. DONNÉES
# ══════════════════════════════════════════════════════════════════
def build_generators(train_dir, test_dir):
    sz, bs = CONFIG["IMG_SIZE"], CONFIG["BATCH_SIZE"]

    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,   # ✅ correct pour EfficientNet
        rotation_range=20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest",
    ).flow_from_directory(train_dir, target_size=sz, batch_size=bs,
                          class_mode="categorical", shuffle=True, seed=42)

    val_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    ).flow_from_directory(test_dir, target_size=sz, batch_size=bs,
                          class_mode="categorical", shuffle=False, seed=42)

    print(f"✅ Train : {train_gen.samples} images | Val : {val_gen.samples} images")
    print(f"✅ Classes : {train_gen.class_indices}")
    return train_gen, val_gen


def get_class_weights(train_gen):
    cw = compute_class_weight("balanced",
                              classes=np.unique(train_gen.classes),
                              y=train_gen.classes)
    cw_dict = dict(enumerate(cw))
    idx = train_gen.class_indices
    # Pondération supplémentaire pour les classes difficiles
    if "glioma"     in idx: cw_dict[idx["glioma"]]     *= 1.5
    if "meningioma" in idx: cw_dict[idx["meningioma"]] *= 1.3
    print(f"⚖️  Class weights : {cw_dict}")
    return cw_dict


# ══════════════════════════════════════════════════════════════════
# 2. FOCAL LOSS
# ══════════════════════════════════════════════════════════════════
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred  = tf.clip_by_value(y_pred, 1e-8, 1.0)
        ce      = -y_true * tf.math.log(y_pred)
        weight  = alpha * y_true * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=1))
    return loss_fn


# ══════════════════════════════════════════════════════════════════
# 3. MODÈLE EfficientNetB0
# ══════════════════════════════════════════════════════════════════
def build_model(num_classes=4):
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*CONFIG["IMG_SIZE"], 3),
    )
    base.trainable = False

    inputs  = tf.keras.Input(shape=(*CONFIG["IMG_SIZE"], 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(CONFIG["DROPOUT"])(x)
    x       = layers.Dense(256, activation="relu",
                           kernel_regularizer=regularizers.l2(CONFIG["L2_REG"]))(x)
    x       = layers.Dropout(CONFIG["DROPOUT"])(x)
    x       = layers.Dense(128, activation="relu",
                           kernel_regularizer=regularizers.l2(CONFIG["L2_REG"]))(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="TumorNet_EfficientNetB0")
    print(f"✅ Modèle construit — {model.count_params():,} paramètres")
    return model, base


# ══════════════════════════════════════════════════════════════════
# 4. AUTO-ENCODEUR
# ══════════════════════════════════════════════════════════════════
def build_autoencoder():
    inputs = tf.keras.Input(shape=(128, 128, 3))
    # Encodeur
    x = layers.Conv2D(32,  3, activation="relu", padding="same", strides=2)(inputs)
    x = layers.Conv2D(64,  3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same", strides=2)(x)
    # Décodeur
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64,  3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(32,  3, activation="relu", padding="same", strides=2)(x)
    outputs = layers.Conv2D(3, 3, activation="sigmoid", padding="same")(x)

    ae = models.Model(inputs, outputs, name="AutoEncoder")
    ae.compile(optimizer="adam", loss="mse")
    return ae


def train_autoencoder(train_dir, ae, epochs=5):
    print("\n🔶 Entraînement Auto-encodeur...")
    raw_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
        train_dir, target_size=(128, 128), batch_size=32,
        class_mode=None, shuffle=True, seed=42)

    X_list = []
    for X in raw_gen:
        X_list.extend(X)
        if len(X_list) >= 500:
            break
    X_arr = np.array(X_list[:500], dtype=np.float32)

    ae.fit(X_arr, X_arr, epochs=epochs, batch_size=32, verbose=1,
           callbacks=[callbacks.EarlyStopping("loss", patience=2,
                                              restore_best_weights=True)])
    print("✅ Auto-encodeur entraîné")
    return ae


def detect_anomaly(ae, img_array, threshold=0.02):
    """Retourne True si l'image est anormale (erreur de reconstruction élevée)"""
    img_resized = tf.image.resize(img_array, (128, 128)).numpy() / 255.0
    img_resized = np.expand_dims(img_resized, 0)
    recon       = ae.predict(img_resized, verbose=0)
    mse         = np.mean((img_resized - recon) ** 2)
    return mse > threshold, float(mse)


# ══════════════════════════════════════════════════════════════════
# 5. GRAD-CAM (XAI)
# ══════════════════════════════════════════════════════════════════
def get_gradcam_heatmap(model, img_array, class_idx=None):
    """Génère une heatmap Grad-CAM pour une image."""
    # Trouver automatiquement la dernière couche Conv2D
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, (layers.Conv2D, tf.keras.layers.Conv2D)):
            last_conv = layer.name
            break

    if last_conv is None:
        # Chercher dans le modèle EfficientNet imbriqué
        for layer in model.layers:
            if hasattr(layer, "layers"):
                for sublayer in reversed(layer.layers):
                    if "conv" in sublayer.name.lower():
                        last_conv = sublayer.name
                        break
            if last_conv:
                break

    if last_conv is None:
        print("⚠️  Grad-CAM : aucune couche Conv2D trouvée")
        return None

    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output]
        )
    except Exception:
        # Essayer avec le sous-modèle EfficientNet
        base_model = [l for l in model.layers if hasattr(l, "layers")][0]
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[base_model.get_layer(last_conv).output, model.output]
        )

    with tf.GradientTape() as tape:
        inputs      = tf.cast(img_array, tf.float32)
        conv_out, preds = grad_model(inputs)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads   = tape.gradient(loss, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam(model, img_path, save_path, class_idx=None):
    """Sauvegarde une image avec heatmap Grad-CAM superposée."""
    from PIL import Image
    img = Image.open(img_path).resize(CONFIG["IMG_SIZE"])
    img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    img_array = img_array[:, :, :3]

    inp      = preprocess_input(np.expand_dims(img_array.astype(np.float32), 0))
    heatmap  = get_gradcam_heatmap(model, inp, class_idx)

    if heatmap is None:
        return

    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = np.array(
        Image.fromarray(heatmap_resized).resize(CONFIG["IMG_SIZE"])
    )
    colormap  = cm.get_cmap("jet")
    heatmap_c = colormap(heatmap_resized / 255.0)[:, :, :3]
    overlay   = (0.6 * img_array / 255.0 + 0.4 * heatmap_c)
    overlay   = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_array); axes[0].set_title("Image originale"); axes[0].axis("off")
    axes[1].imshow(overlay);   axes[1].set_title("Grad-CAM");        axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📸 Grad-CAM sauvegardé → {save_path}")


# ══════════════════════════════════════════════════════════════════
# 6. ENTRAÎNEMENT 2 PHASES
# ══════════════════════════════════════════════════════════════════
def train_model(train_dir, test_dir, save_path=MODEL_PATH):
    train_gen, val_gen = build_generators(train_dir, test_dir)
    cw = get_class_weights(train_gen)
    model, base = build_model(CONFIG["NUM_CLASSES"])
    model.summary()

    # ── Phase 1 : Feature Extraction ──────────────────────────────
    print("\n🔵 Phase 1 : Feature Extraction — EfficientNetB0 gelé")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["LR_FEATURE"]),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    h1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=CONFIG["EPOCHS_FEATURE"], class_weight=cw,
        callbacks=[
            callbacks.EarlyStopping("val_accuracy", patience=5,
                                    restore_best_weights=True, mode="max"),
            callbacks.ReduceLROnPlateau("val_loss", factor=0.5,
                                        patience=3, min_lr=1e-6, verbose=1),
        ], verbose=1,
    )

    # ── Phase 2 : Fine-Tuning ─────────────────────────────────────
    print("\n🟠 Phase 2 : Fine-Tuning — top 30 couches EfficientNetB0")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["LR_FINETUNE"]),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=CONFIG["EPOCHS_FINETUNE"], class_weight=cw,
        callbacks=[
            callbacks.EarlyStopping("val_accuracy", patience=7,
                                    restore_best_weights=True, mode="max"),
            callbacks.ReduceLROnPlateau("val_loss", factor=0.3,
                                        patience=3, min_lr=1e-7, verbose=1),
            callbacks.ModelCheckpoint(save_path, save_best_only=True,
                                      monitor="val_accuracy", mode="max", verbose=1),
        ], verbose=1,
    )
    print(f"\n✅ Modèle sauvegardé → {save_path}")

    # ── Auto-encodeur ─────────────────────────────────────────────
    ae = build_autoencoder()
    ae = train_autoencoder(train_dir, ae, epochs=5)
    ae.save(AE_PATH)
    print(f"✅ Auto-encodeur sauvegardé → {AE_PATH}")

    plot_history(h1, h2)
    return model, ae, val_gen


# ══════════════════════════════════════════════════════════════════
# 7. COURBES D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════
def plot_history(h1, h2):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("TumorNet v10 — EfficientNetB0 (Windows Local)",
                 fontsize=13, fontweight="bold")
    off = len(h1.history["loss"])

    for ax, m, title in zip(axes,
                             ["accuracy", "auc", "recall"],
                             ["Accuracy", "AUC", "Recall"]):
        if m in h1.history:
            e1 = range(1, off + 1)
            ax.plot(e1, h1.history[m],            "#3B82F6", lw=2, label="P1 Train")
            ax.plot(e1, h1.history[f"val_{m}"],   "#3B82F6", lw=2, ls="--", label="P1 Val")
        if m in h2.history:
            e2 = range(off + 1, off + len(h2.history[m]) + 1)
            ax.plot(e2, h2.history[m],            "#F59E0B", lw=2, label="P2 Train")
            ax.plot(e2, h2.history[f"val_{m}"],   "#FCD34D", lw=2, ls="--", label="P2 Val")
        ax.axvline(off, color="gray", ls=":", lw=1.5, label="Fine-Tuning →")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    out = os.path.join(CONFIG["MODELS_DIR"], "training_curves.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"📊 Courbes sauvegardées → {out}")


# ══════════════════════════════════════════════════════════════════
# 8. ÉVALUATION CLINIQUE
# ══════════════════════════════════════════════════════════════════
def evaluate_model(model, val_gen):
    val_gen.reset()
    y_true, y_prob = [], []
    for _ in range(len(val_gen)):
        X, y = next(val_gen)
        y_true.extend(np.argmax(y, axis=1))
        y_prob.extend(model.predict(X, verbose=0))

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    # ── Incertitude ───────────────────────────────────────────────
    max_conf  = np.max(y_prob, axis=1)
    uncertain = np.sum(max_conf < CONFIG["CONF_THRESHOLD"])
    print(f"\n⚠️  Cas incertains (conf < {CONFIG['CONF_THRESHOLD']:.0%}) : "
          f"{uncertain}/{len(y_true)} ({uncertain/len(y_true):.1%})")

    print("\n" + "=" * 60)
    print("     RAPPORT CLINIQUE — TumorNet v10")
    print("=" * 60)
    print(classification_report(y_true, y_pred,
                                target_names=CONFIG["CLASS_NAMES"]))

    f2 = fbeta_score(y_true, y_pred, beta=2, average="weighted")
    print(f"F2-Score  : {f2:.4f}  [cible CDC > 0.90]")

    try:
        yb  = label_binarize(y_true, classes=list(range(CONFIG["NUM_CLASSES"])))
        auc = roc_auc_score(yb, y_prob, multi_class="ovr", average="macro")
        print(f"ROC-AUC   : {auc:.4f}  [cible CDC > 0.95]")
    except Exception as e:
        print(f"ROC-AUC skipped : {e}")

    print("\n📌 Sensibilité par classe (Recall) :")
    for i, name in enumerate(CONFIG["CLASS_NAMES"]):
        mask   = y_true == i
        if mask.sum() == 0:
            continue
        recall = (y_pred[mask] == i).sum() / mask.sum()
        status = "✅" if recall >= 0.85 else "⚠️ "
        print(f"   {status} {name:12s} : {recall:.1%}  ({CONFIG['STADE_MAP'][name]})")

    return y_true, y_pred, y_prob


# ══════════════════════════════════════════════════════════════════
# 9. PRÉDICTION UNIQUE (avec incertitude + anomalie)
# ══════════════════════════════════════════════════════════════════
def predict_single(model, ae, img_path):
    from PIL import Image
    img = Image.open(img_path).convert("RGB").resize(CONFIG["IMG_SIZE"])
    img_array = np.array(img, dtype=np.float32)

    # Détection anomalie
    is_anomaly, mse = detect_anomaly(ae, np.expand_dims(img_array, 0))
    if is_anomaly:
        print(f"⚠️  Image potentiellement anormale (MSE={mse:.4f})")

    # Prédiction
    inp   = preprocess_input(np.expand_dims(img_array, 0))
    probs = model.predict(inp, verbose=0)[0]
    idx   = np.argmax(probs)
    conf  = probs[idx]

    print(f"\n🔬 Résultat : {CONFIG['CLASS_NAMES'][idx]}")
    print(f"   Stade    : {CONFIG['STADE_MAP'][CONFIG['CLASS_NAMES'][idx]]}")
    print(f"   Confiance: {conf:.1%}")
    if conf < CONFIG["CONF_THRESHOLD"]:
        print("   ⚠️  Confiance faible — révision manuelle recommandée")

    print("\n   Probabilités par classe :")
    for i, (name, p) in enumerate(zip(CONFIG["CLASS_NAMES"], probs)):
        bar = "█" * int(p * 20)
        print(f"   {name:12s} : {p:.1%}  {bar}")

    return CONFIG["CLASS_NAMES"][idx], conf, probs


# ══════════════════════════════════════════════════════════════════
# 10. FASTAPI
# ══════════════════════════════════════════════════════════════════
FASTAPI_CODE = '''"""
TumorNet v10 — FastAPI (Windows Local)
Lancer : uvicorn api:app --reload --port 8000
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io, os

app = FastAPI(title="TumorNet v10 API", version="10.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODELS_DIR  = r"C:\\Users\\ACER\\Downloads\\deepfinal\\models"
MODEL_PATH  = os.path.join(MODELS_DIR, "tumor_model.h5")
AE_PATH     = os.path.join(MODELS_DIR, "autoencoder.h5")
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
STADE_MAP   = {"No Tumor": "Stade 0", "Glioma": "Stade II-IV",
               "Meningioma": "Stade I-II", "Pituitary": "Stade I-II"}
THRESHOLD   = 0.70

model, ae = None, None

@app.on_event("startup")
def load_models():
    global model, ae
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"✅ Modèle chargé : {MODEL_PATH}")
    else:
        print(f"❌ Modèle introuvable : {MODEL_PATH}")
    if os.path.exists(AE_PATH):
        ae = tf.keras.models.load_model(AE_PATH, compile=False)
        print(f"✅ Auto-encodeur chargé : {AE_PATH}")

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model":       "loaded" if model else "missing",
        "autoencoder": "loaded" if ae    else "missing",
        "classes":     CLASS_NAMES,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Modèle non chargé")
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    inp = preprocess_input(np.expand_dims(arr, 0))

    probs = model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    name  = CLASS_NAMES[idx]

    # Anomalie
    anomaly, mse = False, 0.0
    if ae:
        small = np.array(img.resize((128, 128)), dtype=np.float32) / 255.0
        recon = ae.predict(np.expand_dims(small, 0), verbose=0)
        mse   = float(np.mean((small - recon[0]) ** 2))
        anomaly = mse > 0.02

    return {
        "class":        name,
        "stade":        STADE_MAP[name],
        "confidence":   round(conf, 4),
        "uncertain":    conf < THRESHOLD,
        "anomaly":      anomaly,
        "anomaly_mse":  round(mse, 6),
        "probabilities": {n: round(float(p), 4)
                          for n, p in zip(CLASS_NAMES, probs)},
    }
'''


# ══════════════════════════════════════════════════════════════════
# 11. DASHBOARD STREAMLIT
# ══════════════════════════════════════════════════════════════════
STREAMLIT_CODE = '''"""
TumorNet v10 — Dashboard Streamlit (Windows Local)
Lancer : streamlit run dashboard.py
"""
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="TumorNet v10", page_icon="🧠", layout="wide")

st.title("🧠 TumorNet v10 — Détection de Tumeurs Cérébrales")
st.markdown("**Modèle :** EfficientNetB0 | **Classes :** Glioma · Meningioma · No Tumor · Pituitary")

API_URL = "http://localhost:8000"

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("API : " + API_URL)
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        h = r.json()
        st.success(f"✅ API connectée")
        st.write(f"Modèle : {h['model']}")
        st.write(f"Auto-encodeur : {h['autoencoder']}")
    except Exception:
        st.error("❌ API non disponible\\nLancer : uvicorn api:app --reload")

# ── Upload ────────────────────────────────────────────────────────
st.header("📤 Charger une IRM cérébrale")
uploaded = st.file_uploader("Format : JPG, PNG", type=["jpg", "jpeg", "png"])

if uploaded:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image chargée")
        img = Image.open(uploaded)
        st.image(img, use_column_width=True)

    with col2:
        st.subheader("🔬 Résultat de l\'analyse")
        with st.spinner("Analyse en cours..."):
            try:
                uploaded.seek(0)
                resp = requests.post(
                    f"{API_URL}/predict",
                    files={"file": ("image.jpg", uploaded.getvalue(), "image/jpeg")},
                    timeout=30,
                )
                result = resp.json()

                # Résultat principal
                color = "🔴" if result["class"] != "No Tumor" else "🟢"
                st.metric("Diagnostic", f"{color} {result[\'class\']}")
                st.metric("Stade",      result["stade"])
                st.metric("Confiance",  f"{result[\'confidence\']:.1%}")

                if result["uncertain"]:
                    st.warning("⚠️ Confiance faible — révision manuelle recommandée")
                if result["anomaly"]:
                    st.error(f"⚠️ Image anormale détectée (MSE={result[\'anomaly_mse\']:.4f})")

                # Probabilités
                st.subheader("📊 Probabilités par classe")
                for cls, prob in result["probabilities"].items():
                    st.progress(prob, text=f"{cls}: {prob:.1%}")

            except Exception as e:
                st.error(f"Erreur API : {e}")
                st.info("Assurez-vous que l\'API tourne : uvicorn api:app --reload")
'''


def save_extra_files():
    """Sauvegarde api.py et dashboard.py"""
    base = r"C:\Users\ACER\Downloads\deepfinal"
    files = {
        os.path.join(base, "api.py"):       FASTAPI_CODE,
        os.path.join(base, "dashboard.py"): STREAMLIT_CODE,
    }
    for path, content in files.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Sauvegardé → {path}")


# ══════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  TumorNet v10 — Windows Local")
    print("=" * 60)
    print(f"📂 Train      : {TRAIN_DIR}")
    print(f"📂 Test       : {TEST_DIR}")
    print(f"🏗️  Backbone   : EfficientNetB0 + Focal Loss")
    print(f"🖥️  TensorFlow : {tf.__version__}")
    print(f"💾 Modèles    : {CONFIG['MODELS_DIR']}")

    # Vérification des dossiers
    for name, path in [("Train", TRAIN_DIR), ("Test", TEST_DIR)]:
        if not os.path.exists(path):
            print(f"❌ Dossier introuvable : {path}")
            print(f"   → Modifiez CONFIG['BASE_DIR'] dans le script")
            sys.exit(1)
        else:
            n = sum(
                len(os.listdir(os.path.join(path, d)))
                for d in os.listdir(path)
                if os.path.isdir(os.path.join(path, d))
            )
            print(f"✅ {name} : {n} images trouvées")

    # Sauvegarde API + Dashboard
    save_extra_files()

    # Entraînement
    model, ae, val_gen = train_model(TRAIN_DIR, TEST_DIR, MODEL_PATH)

    # Évaluation
    evaluate_model(model, val_gen)

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLET")
    print("=" * 60)
    print(f"  📦 Modèle       → {MODEL_PATH}")
    print(f"  📦 Auto-encodeur→ {AE_PATH}")
    print(f"  📊 Courbes      → {CONFIG['MODELS_DIR']}\\training_curves.png")
    print()
    print("  🚀 Pour lancer l'API :")
    print("     cd C:\\Users\\ACER\\Downloads\\deepfinal")
    print("     uvicorn api:app --reload --port 8000")
    print()
    print("  🖥️  Pour lancer le Dashboard :")
    print("     streamlit run dashboard.py")