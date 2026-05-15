import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BASE_URL = os.getenv("APP_BASE_URL", "").strip().rstrip("/")
SECRET = os.getenv("WEBHOOK_SECRET_TOKEN", "").strip()

if not TOKEN:
    print("ERROR: TELEGRAM_BOT_TOKEN is empty")
    sys.exit(1)

if not BASE_URL:
    print("ERROR: APP_BASE_URL is empty. Example: https://your-app.up.railway.app")
    sys.exit(1)

payload = {
    "url": f"{BASE_URL}/webhook",
    "allowed_updates": [
        "business_connection",
        "business_message",
        "edited_business_message",
        "deleted_business_messages",
        "message",
    ],
}

if SECRET:
    payload["secret_token"] = SECRET

response = httpx.post(
    f"https://api.telegram.org/bot{TOKEN}/setWebhook",
    json=payload,
    timeout=30,
)

print(json.dumps(response.json(), ensure_ascii=False, indent=2))
