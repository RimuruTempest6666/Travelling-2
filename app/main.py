from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

from app.ai.gemini import GeminiClient
from app.config import Settings
from app.database import BotDatabase
from app.services.bot_service import BotService
from app.services.safety import SafetyClassifier
from app.telegram.client import TelegramClient


class AppContainer:
    """Simple dependency container for the bot application."""

    def __init__(self) -> None:
        self.settings = Settings.from_env()
        self.database = BotDatabase(self.settings.database_path)
        self.telegram = TelegramClient(self.settings.telegram_bot_token)
        self.gemini = GeminiClient(self.settings.gemini_api_key, self.settings.gemini_model)
        self.safety = SafetyClassifier()
        self.bot_service = BotService(
            settings=self.settings,
            database=self.database,
            telegram=self.telegram,
            gemini=self.gemini,
            safety=self.safety,
        )

    def init(self) -> None:
        self.database.init_schema(self.settings.safe_bot_mode)


container = AppContainer()
app = FastAPI(title="Telegram Business AI Bot", version="2.0.0")


@app.on_event("startup")
async def startup() -> None:
    container.init()


@app.get("/")
async def health() -> dict[str, object]:
    db = container.database
    settings = container.settings
    return {
        "ok": True,
        "service": "telegram-business-ai-bot",
        "version": "2.0.0-oop",
        "mode": db.get_setting("mode", settings.safe_bot_mode),
        "paused": db.get_setting("global_paused", "0") == "1",
        "has_gemini_key": bool(settings.gemini_api_key),
        "gemini_model": settings.gemini_model,
        "owner_id_set": bool(settings.owner_telegram_id),
        "recent_messages_limit": settings.recent_messages_limit,
        "last_ai_error": db.get_setting("last_ai_error", "") or "—",
    }


@app.post("/webhook")
async def webhook(request: Request) -> dict[str, bool]:
    expected_secret = container.settings.webhook_secret_token
    if expected_secret:
        got_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if got_secret != expected_secret:
            raise HTTPException(status_code=403, detail="Invalid webhook secret")

    update = await request.json()
    await container.bot_service.handle_update(update)
    return {"ok": True}
