import os
import re
import json
import base64
import hashlib
import hmac
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, unquote

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# ENV
# =========================================================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

# =========================================================
# APP
# =========================================================
app = FastAPI()

# CORS: поставь сюда домен Netlify (или временно "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # лучше заменить на ["https://dynamic-pavlova-0d09e4.netlify.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Telegram initData verification (эталон)
# =========================================================
def verify_telegram_init_data(init_data: str, bot_token: str) -> Dict[str, Any]:
    if not init_data:
        raise HTTPException(status_code=401, detail="initData empty (WebApp opened not from Telegram button?)")
    if not bot_token:
        raise HTTPException(status_code=500, detail="BOT_TOKEN env is empty on server")

    # 1) Parse query string exactly
    data = dict(parse_qsl(init_data, keep_blank_values=True))

    # 2) Extract hash and remove
    received_hash = data.pop("hash", None)
    if not received_hash:
        raise HTTPException(status_code=401, detail="hash missing in initData")

    # 3) data_check_string
    data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(data.items()))

    # 4) secret_key = sha256(bot_token)
    secret_key = hashlib.sha256(bot_token.encode("utf-8")).digest()

    # 5) calc hash
    calculated_hash = hmac.new(
        secret_key,
        data_check_string.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(calculated_hash, received_hash):
        # Диагностика (чтобы понять, что ломается)
        raise HTTPException(
            status_code=401,
            detail={
                "error": "initData signature invalid",
                "initData_len": len(init_data),
                "initData_head": init_data[:60],
                "received_hash": received_hash,
                "calculated_hash": calculated_hash,
                "data_keys": sorted(list(data.keys()))[:20],  # первые ключи
                "data_check_string_head": data_check_string[:200],
                "BOT_TOKEN_len": len(bot_token),
                "BOT_TOKEN_head": bot_token[:10],
            }
        )

    return data

def get_user_from_verified_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Telegram присылает user как URL-encoded JSON строку.
    ДЕКОДИРУЕМ ТОЛЬКО ПОСЛЕ УСПЕШНОЙ ПРОВЕРКИ ПОДПИСИ.
    """
    user_enc = data.get("user")
    if not user_enc:
        return {}

    try:
        user_json = unquote(user_enc)
        return json.loads(user_json)
    except Exception:
        return {}

# =========================================================
# Helpers: income extraction (заглушка)
# =========================================================
def extract_income_amount_stub(image_bytes: bytes) -> Optional[int]:
    """
    ЗАГЛУШКА.
    Чтобы проверить пайплайн и убрать 401.
    Позже заменишь на свой OCR (pytesseract/cv2 как было).
    """
    # Сейчас просто возвращаем None, чтобы видеть, что подпись прошла.
    # Если хочешь “проверить начисление” — верни фикс. число:
    # return 1000
    return None

# =========================================================
# Routes
# =========================================================
@app.get("/")
def root():
    return {"ok": True, "service": "ozonlegends-backend", "BOT_TOKEN_len": len(BOT_TOKEN)}

@app.post("/api/income-proof")
async def income_proof(
    request: Request,
    file: UploadFile = File(...),
    initData: Optional[str] = Form(None),
):
    # 0) Достаём initData также из заголовка (если вдруг фронт так отправляет)
    header_init = request.headers.get("x-tg-init-data") or request.headers.get("X-Tg-Init-Data")
    final_init = initData or header_init

    # Диагностика входа
    if not final_init:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "initData not provided",
                "hint": "Open WebApp from Telegram button; send tg.initData",
                "got_form_initData": bool(initData),
                "got_header_initData": bool(header_init),
            }
        )

    # 1) Проверяем подпись Telegram
    verified = verify_telegram_init_data(final_init, BOT_TOKEN)

    # 2) Достаём юзера
    user = get_user_from_verified_data(verified)
    telegram_id = user.get("id")
    name = user.get("username") or user.get("first_name") or "Игрок"

    if not telegram_id:
        raise HTTPException(status_code=401, detail={"error": "user.id missing after verify", "user": user})

    # 3) Читаем файл
    img_bytes = await file.read()
    if not img_bytes or len(img_bytes) < 1000:
        raise HTTPException(status_code=422, detail="File is too small or empty")

    # 4) Распознаём сумму (ПОКА ЗАГЛУШКА)
    amount = extract_income_amount_stub(img_bytes)

    # Если хочешь прямо сейчас проверить начисление — временно раскомментируй:
    # amount = 1000

    if amount is None:
        # Подпись прошла, но OCR пока не настроен
        return {
            "decision": "rejected",
            "reason": "OCR not configured yet (signature OK)",
            "debug": {
                "telegram_id": telegram_id,
                "name": name,
                "file_name": file.filename,
                "file_size": len(img_bytes),
                "initData_len": len(final_init),
            }
        }

    # 5) Диапазон суммы 10..20000
    if not (10 <= amount <= 20000):
        return {
            "decision": "rejected",
            "reason": f"Amount out of range: {amount} (need 10..20000)",
            "amount": amount,
        }

    # 6) Здесь дальше должна быть твоя логика:
    # - антифрод (хеш файла, лимит на день, дедупликация)
    # - начисление XP/coins
    # - сохранение state в базе
    # Пока вернём пример успешного ответа, чтобы фронт обновился
    reward = {"xp": amount // 100, "coins": amount // 250}

    # Пример "state" (замени на твой реальный state из БД)
    state = {
        "meta": {"version": 1},
        "player": {"id": str(telegram_id), "name": name},
        "level": 1,
        "xp": reward["xp"],
        "coins": reward["coins"],
        "incomeTotal": amount,
        "day": {"date": "2026-01-08", "income": amount, "shifts": 1},
        "character": {"speed": 1, "accuracy": 1, "power": 1},
        "warehouse": {"capacity": 1, "automation": 1, "comfort": 1},
        "energy": {"current": 10, "max": 10, "lastResetDate": "2026-01-08"},
    }

    return {
        "decision": "accepted",
        "amount": amount,
        "reward": reward,
        "state": state,
        "debug": {
            "telegram_id": telegram_id,
            "name": name,
            "file_size": len(img_bytes),
        }
    }
