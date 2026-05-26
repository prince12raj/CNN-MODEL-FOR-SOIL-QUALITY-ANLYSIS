import gradio as gr
import numpy as np
import json
import os
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io

print("Loading CNN model...")
model = tf.keras.models.load_model("soil_cnn_model.h5")
print("✅ Model loaded!")

with open("soil_info.json", "r") as f:
    soil_info = json.load(f)

CLASS_NAMES = [
    "Alluvial_Soil", "Arid_Soil", "Black_Soil",
    "Laterite_Soil", "Mountain_Soil", "Red_Soil", "Yellow_Soil"
]
IMG_SIZE = (128, 128)


def analyze_soil(image):
    if image is None:
        return {"error": "No image provided"}
    try:
        img  = Image.fromarray(image).convert("RGB").resize(IMG_SIZE)
        arr  = np.array(img) / 255.0
        arr  = np.expand_dims(arr, axis=0)
        preds     = model.predict(arr, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        soil_type = CLASS_NAMES[class_idx]
        confidence = float(np.max(preds)) * 100
        all_scores = {CLASS_NAMES[i]: round(float(preds[i])*100, 2) for i in range(len(CLASS_NAMES))}
        info = soil_info.get(soil_type, {})
        return {
            "success":    True,
            "soil_type":  soil_type,
            "confidence": round(confidence, 2),
            "crops":      info.get("crops", []),
            "water":      info.get("water", ""),
            "ph":         info.get("ph", ""),
            "nutrients":  info.get("nutrients", []),
            "all_scores": all_scores
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def predict_ui(image):
    result = analyze_soil(image)
    if not result.get("success"):
        return f"❌ Error: {result.get('error')}"
    soil   = result["soil_type"].replace("_", " ")
    conf   = result["confidence"]
    crops  = ", ".join(result["crops"])
    scores = "\n".join([f"  {k.replace('_',' ')}: {v}%" for k, v in sorted(result["all_scores"].items(), key=lambda x: -x[1])])
    return f"""🌱 SOIL ANALYSIS REPORT
═══════════════════════════
🏆 Soil Type   : {soil}
📊 Confidence  : {conf:.2f}%
💧 Water       : {result['water']}
🧪 pH Range    : {result['ph']}

🌾 Recommended Crops
{crops}

🧬 Key Nutrients
{", ".join(result['nutrients'])}

📈 All Scores
{scores}"""


# ── Gradio UI ──
demo = gr.Interface(
    fn=predict_ui,
    inputs=gr.Image(label="Upload Soil Image", type="numpy"),
    outputs=gr.Textbox(label="Soil Analysis Result", lines=25),
    title="🌾 SMART Agriculture — Soil Classifier",
    description="Upload a soil image to get soil type, crops, and nutrient recommendations.",
    theme=gr.themes.Soft(primary_hue="green")
)

# ── FastAPI JSON endpoint ──
fastapi_app = FastAPI()

@fastapi_app.get("/")
def root():
    return {"status": "ok", "message": "Soil CNN API running"}

@fastapi_app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    try:
        contents  = await file.read()
        img       = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(img)
        result    = analyze_soil(img_array)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"success": False, "error": str(e)}, status_code=500)

app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")