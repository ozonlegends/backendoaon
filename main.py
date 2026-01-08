import base64
import hashlib
import hmac
import os
import re
from datetime import datetime, date
from typing import Any, Dict, Optional, Tuple, List

import httpx
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2

# =========================
# ENV
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Коридор суммы (по твоей корректировке)
MIN_AMOUNT = int(os.getenv("MIN_AMOUNT", "10"))
MAX_AMOUNT = int(os.getenv("MAX_AMOUNT", "20000"))

# OCR и антифрод
OCR_MIN_CONF = float(os.getenv("OCR_MIN_CONF", "0.55"))  # если ниже -> reject
DAILY_ACCEPT_LIMIT = int(os.getenv("DAILY_ACCEPT_LIMIT", "2"))  # accepted в сутки
KEYWORDS = os.getenv(
    "OCR_KEYWORDS",
    "выплаты,заработано,доход,перевод,начислено,можно вывести"
).split(",")

# Таймзона дня (без UTC-ошибок)
# Для простоты используем local date сервера. На Render можно поставить TZ=Europe/Moscow.
def today_local() -> date:
    return datetime.now().date()

# =========================
# Telegram initData verify
# =========================
def parse_init_data(init_data: str) -> Dict[str, str]:
    # initData выглядит как querystring: "query_id=...&user=...&hash=..."
    parts = init_data.split("&")
    kv = {}
    for p in parts:
        if not p:
            continue
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kv[k] = v
    return kv

def build_data_check_string(kv: Dict[str, str]) -> str:
    # исключаем hash
    items = [(k, kv[k]) for k in kv.keys() if k != "hash"]
    items.sort(key=lambda x: x[0])
    return "\n".join([f"{k}={v}" for k, v in items])

def verify_telegram_init_data(init_data: str, bot_token: str) -> Dict[str, Any]:
    if not init_data or not bot_token:
        raise HTTPException(status_code=401, detail="initData или BOT_TOKEN отсутствует")

    kv = parse_init_data(init_data)
    given_hash = kv.get("hash")
    if not given_hash:
        raise HTTPException(status_code=401, detail="hash отсутствует в initData")

    dcs = build_data_check_string(kv)

    secret_key = hashlib.sha256(bot_token.encode("utf-8")).digest()
    calc_hash = hmac.new(secret_key, dcs.encode("utf-8"), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(calc_hash, given_hash):
        raise HTTPException(status_code=401, detail="initData подпись невалидна")

    # user приходит URL-encoded JSON
    # В initDataUnsafe.user: {id, username, first_name, ...}
    # Здесь kv["user"] URL-encoded; Telegram присылает percent-encoding.
    # FastAPI уже не декодирует это автоматически. Декодируем сами:
    from urllib.parse import unquote
    user_json = kv.get("user")
    if not user_json:
        raise HTTPException(status_code=401, detail="user отсутствует в initData")

    import json
    user = json.loads(unquote(user_json))
    if "id" not in user:
        raise HTTPException(status_code=401, detail="user.id отсутствует")

    return user

# =========================
# Supabase REST helpers
# =========================
def _sb_headers() -> Dict[str, str]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=500, detail="Supabase ENV не настроен")
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

async def sb_select_one(table: str, eq: Dict[str, Any], fields: str = "*") -> Optional[Dict[str, Any]]:
    # GET /rest/v1/{table}?col=eq.value&select=...
    params = {"select": fields}
    for k, v in eq.items():
        params[k] = f"eq.{v}"
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params, headers=_sb_headers())
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"Supabase select error: {r.text}")
        rows = r.json()
        return rows[0] if rows else None

async def sb_select(table: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=params, headers=_sb_headers())
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"Supabase select error: {r.text}")
        return r.json() or []

async def sb_upsert(table: str, payload: Dict[str, Any], on_conflict: str) -> Dict[str, Any]:
    url = f"{SUPABASE_URL}/rest/v1/{table}?on_conflict={on_conflict}"
    headers = _sb_headers()
    headers["Prefer"] = "resolution=merge-duplicates,return=representation"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"Supabase upsert error: {r.text}")
        rows = r.json()
        return rows[0] if rows else payload

async def sb_insert(table: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json=payload, headers=_sb_headers())
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail=f"Supabase insert error: {r.text}")
        rows = r.json()
        return rows[0] if rows else payload

# =========================
# Game mechanics (как у тебя в JS)
# =========================
def xp_need_for_level(lvl: int) -> int:
    return int(100 + (lvl - 1) * 40 + ((lvl - 1) ** 2) * 6)

def income_to_xp(income: int, speed: int, capacity: int) -> int:
    base = income / 100
    bonus = 1 + (speed - 1) * 0.06 + (capacity - 1) * 0.05
    return int(base * bonus)

def income_to_coins(income: int, accuracy: int, automation: int) -> int:
    base = income / 250
    bonus = 1 + (accuracy - 1) * 0.06 + (automation - 1) * 0.05
    return int(base * bonus)

def recalc_energy_max(comfort: int) -> int:
    return 10 + (comfort - 1) * 2

def apply_level_ups(level: int, xp: int) -> Tuple[int, int, int]:
    # returns: new_level, new_xp, levels_gained
    gained = 0
    need = xp_need_for_level(level)
    while xp >= need:
        xp -= need
        level += 1
        gained += 1
        need = xp_need_for_level(level)
    return level, xp, gained

def ensure_day(player: Dict[str, Any]) -> None:
    t = today_local()
    if str(player.get("day_date")) != str(t):
        player["day_date"] = str(t)
        player["income_today"] = 0
        player["shifts_today"] = 0

    # энергия раз в день
    if str(player.get("energy_last_reset_date")) != str(t):
        player["energy_last_reset_date"] = str(t)
        player["energy_current"] = int(player.get("energy_max") or 10)

# =========================
# OCR
# =========================
def preprocess_image_for_ocr(pil_img: Image.Image) -> np.ndarray:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # легкая нормализация
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # повысим контраст
    gray = cv2.equalizeHist(gray)

    # бинаризация
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return th

def ocr_text_and_confidence(pil_img: Image.Image) -> Tuple[str, float]:
    img = preprocess_image_for_ocr(pil_img)
    data = pytesseract.image_to_data(
        img,
        lang="rus+eng",
        output_type=pytesseract.Output.DICT
    )
    texts = []
    confs = []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        if txt and txt.strip():
            texts.append(txt.strip())
        try:
            c = float(conf)
            if c >= 0:
                confs.append(c)
        except:
            pass
    full_text = " ".join(texts)
    avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return full_text, avg_conf

def extract_amount(text: str) -> Optional[int]:
    # Ищем числа вида 12 345 или 12345 или 12,345
    # и берём "самое правдоподобное" (в пределах диапазона)
    candidates = []

    # заменим неразрывные пробелы
    t = text.replace("\u00A0", " ").lower()

    # regex: группа цифр с пробелами/запятыми/точками между тысячами
    for m in re.finditer(r"(\d[\d\s\.,]{0,10}\d)", t):
        raw = m.group(1)
        cleaned = re.sub(r"[^\d]", "", raw)
        if not cleaned:
            continue
        try:
            val = int(cleaned)
            candidates.append(val)
        except:
            continue

    # фильтруем по диапазону
    in_range = [c for c in candidates if MIN_AMOUNT <= c <= MAX_AMOUNT]
    if not in_range:
        return None

    # если несколько — берём максимальную в диапазоне
    return max(in_range)

def has_keywords(text: str) -> bool:
    t = text.lower()
    hits = 0
    for kw in KEYWORDS:
        kw = kw.strip().lower()
        if kw and kw in t:
            hits += 1
    return hits >= 1

# =========================
# Anti-fraud checks
# =========================
async def check_daily_limit(telegram_id: int) -> bool:
    # сколько accepted сегодня
    t = today_local()
    rows = await sb_select(
        "income_proofs",
        {
            "select": "id,created_at,decision",
            "telegram_id": f"eq.{telegram_id}",
            "decision": "eq.accepted",
            "created_at": f"gte.{t.isoformat()}T00:00:00",
        }
    )
    return len(rows) < DAILY_ACCEPT_LIMIT

# =========================
# API
# =========================
app = FastAPI()

# Разрешаем запросы с твоего домена (Netlify) и локально
# Можно поставить строго: ORIGINS="https://...netlify.app"
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/api/income-proof")
async def income_proof(
    initData: str = Form(...),
    file: UploadFile = File(...)
):
    user = verify_telegram_init_data(initData, BOT_TOKEN)
    telegram_id = int(user["id"])
    name = user.get("username")
    if name:
        display_name = "@" + name
    else:
        display_name = user.get("first_name") or "Игрок"

    # читаем файл
    content = await file.read()
    if not content or len(content) < 2000:
        raise HTTPException(status_code=400, detail="Файл пустой или слишком маленький")

    img_hash = hashlib.sha256(content).hexdigest()

    # проверка повтора (идемпотентность)
    existing = await sb_select_one("income_proofs", {"image_hash": img_hash}, fields="id,decision,ocr_amount")
    if existing:
        return {
            "decision": "rejected",
            "reason": "Этот скрин уже был засчитан ранее",
            "amount": existing.get("ocr_amount"),
        }

    # лимит в сутки
    if not await check_daily_limit(telegram_id):
        return {
            "decision": "rejected",
            "reason": f"Достигнут лимит {DAILY_ACCEPT_LIMIT} засчитанных скринов в сутки",
        }

    # OCR
    try:
        pil = Image.open(io_bytes_to_stream(content))
    except Exception:
        # fallback: Pillow иногда требует BytesIO
        from io import BytesIO
        pil = Image.open(BytesIO(content))

    text, conf = ocr_text_and_confidence(pil)
    amount = extract_amount(text)

    # проверки
    if conf < OCR_MIN_CONF:
        decision = "rejected"
        reason = f"Низкое качество распознавания (conf={conf:.2f}). Сделай более чёткий скрин."
    elif not has_keywords(text):
        decision = "rejected"
        reason = "Не найдено ключевых слов выплат/дохода на скрине"
    elif amount is None:
        decision = "rejected"
        reason = f"Не удалось распознать сумму в диапазоне {MIN_AMOUNT}-{MAX_AMOUNT}"
    else:
        decision = "accepted"
        reason = None

    # создаём/обновляем игрока
    player = await sb_select_one("players", {"telegram_id": telegram_id}, fields="*")
    if not player:
        player = {
            "telegram_id": telegram_id,
            "name": display_name,
            "level": 1, "xp": 0, "coins": 0,
            "income_total": 0, "income_today": 0, "shifts_today": 0,
            "day_date": str(today_local()),
            "character_speed": 1, "character_accuracy": 1, "character_power": 1,
            "warehouse_capacity": 1, "warehouse_automation": 1, "warehouse_comfort": 1,
            "energy_max": 10, "energy_current": 10, "energy_last_reset_date": str(today_local())
        }
        player = await sb_upsert("players", player, on_conflict="telegram_id")

    ensure_day(player)

    reward_xp = 0
    reward_coins = 0

    if decision == "accepted":
        income = int(amount)

        # применяем твою механику
        gained_xp = income_to_xp(income, int(player["character_speed"]), int(player["warehouse_capacity"]))
        gained_coins = income_to_coins(income, int(player["character_accuracy"]), int(player["warehouse_automation"]))

        # обновляем агрегаты
        player["income_total"] = int(player.get("income_total") or 0) + income
        player["income_today"] = int(player.get("income_today") or 0) + income
        player["shifts_today"] = int(player.get("shifts_today") or 0) + 1

        player["xp"] = int(player.get("xp") or 0) + int(gained_xp)
        player["coins"] = int(player.get("coins") or 0) + int(gained_coins)

        new_energy_max = recalc_energy_max(int(player["warehouse_comfort"]))
        player["energy_max"] = int(new_energy_max)
        if int(player["energy_current"]) > int(player["energy_max"]):
            player["energy_current"] = int(player["energy_max"])

        # level-ups
        lvl, xp_left, levels_gained = apply_level_ups(int(player["level"]), int(player["xp"]))
        player["level"] = int(lvl)
        player["xp"] = int(xp_left)

        reward_xp = int(gained_xp)
        reward_coins = int(gained_coins)

        player["updated_at"] = datetime.utcnow().isoformat()

        # записываем игрока
        player = await sb_upsert("players", player, on_conflict="telegram_id")

    # пишем proof в БД (и rejected тоже — чтобы видеть попытки)
    await sb_insert("income_proofs", {
        "telegram_id": telegram_id,
        "image_hash": img_hash,
        "ocr_amount": int(amount) if amount is not None else None,
        "ocr_text": text[:4000],
        "confidence": float(conf),
        "decision": decision,
        "reason": reason,
        "reward_xp": reward_xp,
        "reward_coins": reward_coins
    })

    # отдадим state в формате, близком твоему JS
    return {
        "decision": decision,
        "reason": reason,
        "amount": int(amount) if amount is not None else None,
        "confidence": round(conf, 3),
        "reward": {"xp": reward_xp, "coins": reward_coins},
        "state": player_to_state(player)
    }

@app.get("/api/leaderboard")
async def leaderboard(limit: int = 50):
    limit = max(1, min(limit, 50))
    rows = await sb_select(
        "players",
        {
            "select": "telegram_id,name,income_total,level,updated_at",
            "order": "income_total.desc",
            "limit": str(limit)
        }
    )
    return {"rows": rows}

# =========================
# helpers
# =========================
def player_to_state(p: Dict[str, Any]) -> Dict[str, Any]:
    # приводим к твоему state-формату (ядро)
    t = today_local().isoformat()
    return {
        "meta": {"version": 1},
        "player": {"id": str(p["telegram_id"]), "name": p.get("name") or "Игрок"},
        "level": int(p.get("level") or 1),
        "xp": int(p.get("xp") or 0),
        "coins": int(p.get("coins") or 0),
        "incomeTotal": int(p.get("income_total") or 0),
        "day": {
            "date": str(p.get("day_date") or t),
            "income": int(p.get("income_today") or 0),
            "shifts": int(p.get("shifts_today") or 0),
        },
        "character": {
            "speed": int(p.get("character_speed") or 1),
            "accuracy": int(p.get("character_accuracy") or 1),
            "power": int(p.get("character_power") or 1),
        },
        "warehouse": {
            "capacity": int(p.get("warehouse_capacity") or 1),
            "automation": int(p.get("warehouse_automation") or 1),
            "comfort": int(p.get("warehouse_comfort") or 1),
        },
        "energy": {
            "current": int(p.get("energy_current") or 10),
            "max": int(p.get("energy_max") or 10),
            "lastResetDate": str(p.get("energy_last_reset_date") or t),
        }
    }

def io_bytes_to_stream(content: bytes):
    from io import BytesIO
    return BytesIO(content)
