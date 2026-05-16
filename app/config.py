from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    gemini_api_key: str
    gemini_model: str
    owner_telegram_id: int
    webhook_secret_token: str
    bot_mode: str
    database_path: Path
    prompt_path: Path
    knowledge_path: Path
    recent_messages_limit: int
    search_messages_limit: int
    confidence_threshold: float
    manual_takeover_minutes: int
    send_handoff_to_chat: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            gemini_api_key=(os.getenv("GEMINI_API_KEY", "") or os.getenv("GEMINI_KEY", "")).strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip(),
            owner_telegram_id=int(os.getenv("OWNER_TELEGRAM_ID", "0") or 0),
            webhook_secret_token=os.getenv("WEBHOOK_SECRET_TOKEN", "").strip(),
            bot_mode=os.getenv("BOT_MODE", "draft").strip().lower(),
            database_path=Path(os.getenv("DATABASE_PATH", "data/bot.sqlite3")),
            prompt_path=Path(os.getenv("PROMPT_PATH", "data/PROMPT.md")),
            knowledge_path=Path(os.getenv("KNOWLEDGE_PATH", "data/KNOWLEDGE.md")),
            recent_messages_limit=int(os.getenv("RECENT_MESSAGES_LIMIT", "40")),
            search_messages_limit=int(os.getenv("SEARCH_MESSAGES_LIMIT", "12")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.35")),
            manual_takeover_minutes=int(os.getenv("MANUAL_TAKEOVER_MINUTES", "30")),
            send_handoff_to_chat=os.getenv("SEND_HANDOFF_TO_CHAT", "0").strip() == "1",
        )

    @property
    def safe_bot_mode(self) -> str:
        return self.bot_mode if self.bot_mode in {"auto", "draft", "silent"} else "draft"
