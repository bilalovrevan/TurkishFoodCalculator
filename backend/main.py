from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io, json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# LOAD CLASSES
# =============================
with open("classes.txt", "r", encoding="utf-8") as f:
    CLASSES = [c.strip() for c in f.readlines()]

# normalize helper
def norm(name: str):
    return name.lower().replace(" ", "_")

# =============================
# LOAD NUTRITION DATABASE
# =============================
with open("nutrition.json", "r", encoding="utf-8") as f:
    RAW_NUTRITION = json.load(f)

NUTRITION_DB = { norm(k): v for k, v in RAW_NUTRITION.items() }

# =============================
# AUTO ESTIMATION (fallback)
# =============================
def estimate_nutrition(food_name: str):
    name = food_name.lower()

    if "salad" in name or "salata" in name:
        return { "calories": 120, "protein": 3, "fat": 5, "carbs": 10 }

    if "soup" in name or "corba" in name:
        return { "calories": 150, "protein": 6, "fat": 4, "carbs": 18 }

    if "dessert" in name or "tatli" in name:
        return { "calories": 380, "protein": 5, "fat": 18, "carbs": 45 }

    if "kebab" in name or "doner" in name:
        return { "calories": 480, "protein": 32, "fat": 28, "carbs": 20 }

    # generic fallback
    return { "calories": 300, "protein": 15, "fat": 15, "carbs": 30 }

# =============================
# MODEL (mock – sən özünü qoy)
# =============================
CONF_THRESHOLD = 0.40

@app.post("/predict")
async def predict(file: UploadFile = File(...), grams: int = 180):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 🔴 BURDA SƏNİN REAL MODEL OLACAQ
        # example:
        food_name = "Hunkar_Begendi"
        confidence = 0.73

        if confidence < CONF_THRESHOLD:
            return {
                "success": False,
                "message": "This is not a food image",
                "confidence": confidence
            }

        key = norm(food_name)

        base = NUTRITION_DB.get(key)

        # if not found → auto estimate
        if base is None:
            base = estimate_nutrition(food_name)

        factor = grams / 180

        nutrition = {
            "calories": round(base["calories"] * factor, 1),
            "protein": round(base["protein"] * factor, 1),
            "fat": round(base["fat"] * factor, 1),
            "carbs": round(base["carbs"] * factor, 1)
        }

        return {
            "success": True,
            "food_name": food_name,
            "confidence": confidence,
            "nutrition": nutrition
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }
