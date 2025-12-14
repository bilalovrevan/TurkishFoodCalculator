from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io, os, json

from storage import ensure_dir, load_json, atomic_write_json, today_str, now_iso, new_id

# ================= APP =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF_THRESHOLD = 0.45
TOP_K = 3

TURKISH_CLASSES = {
    "cig kofte", "yaprak sarma", "hunkar begendi",
    "iskender", "manti", "lahmacun", "doner",
    "adana kebap", "tantuni", "menemen", "kisir"
}

# ================= PATH HELPERS =================
def pick_path(*candidates: str) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Missing file. Tried: " + " | ".join(candidates))

def norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def pretty(name: str) -> str:
    # "_" və "-" boşluğa, çoxlu boşluqları təmizlə
    s = name.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    # Turkish class adları Title Case daha yaxşı görünür
    return s.title()

# ================= PROJECT PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "backend_data")  # api.py harda olsa da burda saxlayır
ensure_dir(DATA_DIR)

HISTORY_PATH = os.path.join(DATA_DIR, "history.json")
GOALS_PATH = os.path.join(DATA_DIR, "goals.json")

MODEL_PATH = pick_path(
    os.path.join(BASE_DIR, "model.pth"),
    os.path.join(BASE_DIR, "models", "best_foodvision_model.pth"),
    os.path.join(BASE_DIR, "backend", "model.pth"),
    os.path.join(BASE_DIR, "backend", "models", "best_foodvision_model.pth"),
)

CLASSES_PATH = pick_path(
    os.path.join(BASE_DIR, "classes.txt"),
    os.path.join(BASE_DIR, "models", "classes.txt"),
    os.path.join(BASE_DIR, "backend", "classes.txt"),
    os.path.join(BASE_DIR, "backend", "models", "classes.txt"),
)

NUTRITION_PATH = pick_path(
    os.path.join(BASE_DIR, "nutrition.json"),
    os.path.join(BASE_DIR, "backend", "nutrition.json"),
    os.path.join(BASE_DIR, "backend", "backend", "nutrition.json"),
)

# ================= LOAD CLASSES + NUTRITION =================
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASSES = [x.strip() for x in f if x.strip()]

with open(NUTRITION_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

NUTRITION_DB = {norm(k): v for k, v in raw.items()}

# ================= LOAD MODEL =================
state = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(state, dict) and "state_dict" in state:
    state_dict = state["state_dict"]
else:
    state_dict = state

num_classes = state_dict["fc.weight"].shape[0]

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

if len(CLASSES) != num_classes:
    print(f"[WARN] class mismatch: model={num_classes} classes.txt={len(CLASSES)}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ================= FALLBACK NUTRITION =================
def estimate_nutrition(food_name: str):
    n = food_name.lower()
    if "salad" in n or "salata" in n:
        return {"calories": 120, "protein": 3, "fat": 5, "carbs": 10}
    if "soup" in n or "corba" in n:
        return {"calories": 150, "protein": 6, "fat": 4, "carbs": 18}
    if "tatli" in n or "cake" in n or "baklava" in n:
        return {"calories": 380, "protein": 5, "fat": 18, "carbs": 45}
    if "kebab" in n or "doner" in n:
        return {"calories": 480, "protein": 32, "fat": 28, "carbs": 20}
    return {"calories": 300, "protein": 15, "fat": 15, "carbs": 30}

def scale_nutrition(base: Dict[str, Any], grams: int, base_grams: int = 180):
    factor = float(grams) / float(base_grams)
    return {
        "calories": round(float(base.get("calories", 0)) * factor, 1),
        "protein":  round(float(base.get("protein", 0))  * factor, 1),
        "fat":      round(float(base.get("fat", 0))      * factor, 1),
        "carbs":    round(float(base.get("carbs", 0))    * factor, 1),
    }

# ================= STORAGE (history + goals) =================
def get_goals():
    # default goals
    default = {
        "daily_calories": 2200,
        "daily_protein": 130,
        "daily_fat": 70,
        "daily_carbs": 250
    }
    return load_json(GOALS_PATH, default)

def save_goals(g):
    atomic_write_json(GOALS_PATH, g)

def get_history():
    return load_json(HISTORY_PATH, [])

def save_history(h):
    atomic_write_json(HISTORY_PATH, h)

def summarize_for_date(d: str):
    items = [x for x in get_history() if x.get("date") == d]
    total = {"calories": 0.0, "protein": 0.0, "fat": 0.0, "carbs": 0.0}
    for it in items:
        total["calories"] += float(it.get("calories", 0) or 0)
        total["protein"]  += float(it.get("protein", 0) or 0)
        total["fat"]      += float(it.get("fat", 0) or 0)
        total["carbs"]    += float(it.get("carbs", 0) or 0)
    # 1 decimal
    total = {k: round(v, 1) for k, v in total.items()}
    return {"date": d, "items": items, "total": total, "goals": get_goals()}

# ================= MODELS =================
class GoalsIn(BaseModel):
    daily_calories: Optional[float] = 2200
    daily_protein: Optional[float] = 130
    daily_fat: Optional[float] = 70
    daily_carbs: Optional[float] = 250

class MealLogIn(BaseModel):
    food_name: str
    grams: int
    confidence: Optional[float] = None
    calories: float
    protein: float
    fat: float
    carbs: float
    meal_type: Optional[str] = "meal"  # breakfast/lunch/dinner/snack/meal
    date: Optional[str] = None         # YYYY-MM-DD

# ================= AI PREDICT =================
@app.post("/predict")
async def predict(file: UploadFile = File(...), grams: int = Form(180)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, TOP_K)

    candidates = []
    for p, i in zip(top_probs, top_idxs):
        raw_name = CLASSES[int(i)]
        candidates.append({
            "food_name": pretty(raw_name),
            "confidence": round(float(p.item()), 4)
        })

    best_name = candidates[0]["food_name"]
    best_conf = candidates[0]["confidence"]

    # Turkish boost
    if best_name.lower() in TURKISH_CLASSES:
        best_conf = min(best_conf * 1.08, 1.0)

    # reject low confidence
    if best_conf < CONF_THRESHOLD:
        return {
            "food_detected": False,
            "confidence": round(best_conf, 4),
            "grams": int(grams),
            "candidates": candidates
        }

    key = norm(best_name)
    base = NUTRITION_DB.get(key)
    nutrition_available = True
    if base is None:
        base = estimate_nutrition(best_name)
        nutrition_available = False

    scaled = scale_nutrition(base, grams, base_grams=180)

    return {
        "food_detected": True,
        "food_name": best_name,
        "confidence": round(best_conf, 4),
        "grams": int(grams),
        "nutrition_available": nutrition_available,
        **scaled,
        "candidates": candidates
    }

# ================= GOALS ENDPOINTS =================
@app.get("/goals")
def read_goals():
    return get_goals()

@app.post("/goals")
def update_goals(payload: GoalsIn):
    g = get_goals()
    # update only provided
    data = payload.model_dump(exclude_none=True)
    g.update(data)
    # sanitize
    for k in ["daily_calories", "daily_protein", "daily_fat", "daily_carbs"]:
        g[k] = float(g.get(k, 0) or 0)
    save_goals(g)
    return {"ok": True, "goals": g}

# ================= HISTORY ENDPOINTS =================
@app.post("/log-meal")
def log_meal(payload: MealLogIn):
    h = get_history()

    d = payload.date or today_str()
    entry = {
        "id": new_id(),
        "created_at": now_iso(),
        "date": d,
        "meal_type": payload.meal_type or "meal",
        "food_name": pretty(payload.food_name),
        "grams": int(payload.grams),
        "confidence": float(payload.confidence) if payload.confidence is not None else None,
        "calories": round(float(payload.calories), 1),
        "protein": round(float(payload.protein), 1),
        "fat": round(float(payload.fat), 1),
        "carbs": round(float(payload.carbs), 1),
    }

    h.append(entry)
    save_history(h)
    return {"ok": True, "item": entry}

@app.get("/history")
def read_history(date: Optional[str] = None):
    d = date or today_str()
    return summarize_for_date(d)

@app.delete("/history/{item_id}")
def delete_history_item(item_id: str):
    h = get_history()
    new_h = [x for x in h if x.get("id") != item_id]
    save_history(new_h)
    return {"ok": True, "deleted": item_id}

@app.get("/summary/today")
def summary_today():
    return summarize_for_date(today_str())

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": str(DEVICE),
        "model_path": MODEL_PATH,
        "classes": len(CLASSES),
        "data_dir": DATA_DIR
    }
