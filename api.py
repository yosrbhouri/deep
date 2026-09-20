"""
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
import cv2
import base64
from fastapi.responses import JSONResponse

app = FastAPI(title="TumorNet v10 API", version="10.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODELS_DIR  = r"C:\Users\ACER\Downloads\deepfinal\models"
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

@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # 📥 Lire image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))

    # 🔄 Préprocessing
    img_array = np.array(img, dtype=np.float32)
    img_preprocessed = preprocess_input(img_array.copy())
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    # 🔍 Trouver la dernière couche Conv2D
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    if last_conv_layer_name is None:
        raise HTTPException(status_code=500, detail="Aucune couche Conv2D trouvée")

    # 🧠 Construire le modèle Grad-CAM (FIX PRINCIPAL)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,  # ✅ IMPORTANT (corrige ton erreur)
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # 🎯 Calcul Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 📈 Gradients
    grads = tape.gradient(class_channel, conv_outputs)

    # 📊 Moyenne des gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 🔥 Heatmap
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 🧹 Normalisation
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # 🎨 Resize + Color
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    # 🖼️ Superposition
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)

    # 📦 Encodage base64
    _, buffer = cv2.imencode(".png", overlay)
    gradcam_b64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({"gradcam_base64": gradcam_b64})