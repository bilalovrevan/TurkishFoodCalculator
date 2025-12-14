import os, json, uuid
from datetime import datetime, date

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def atomic_write_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def today_str() -> str:
    return date.today().isoformat()

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def new_id() -> str:
    return str(uuid.uuid4())
