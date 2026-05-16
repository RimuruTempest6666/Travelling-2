from __future__ import annotations

from typing import Any

from app.models import BusinessMessage


class TelegramUpdateParser:
    @staticmethod
    def get_update_type(update: dict[str, Any]) -> str:
        if "business_connection" in update:
            return "business_connection"
        if "business_message" in update:
            return "business_message"
        if "edited_business_message" in update:
            return "edited_business_message"
        if "deleted_business_messages" in update:
            return "deleted_business_messages"
        if "message" in update:
            return "message"
        return "unknown"

    @staticmethod
    def parse_business_message(update: dict[str, Any]) -> BusinessMessage | None:
        message = update.get("business_message")
        if not isinstance(message, dict):
            return None

        chat = message.get("chat") or {}
        user = message.get("from") or {}
        business_connection_id = message.get("business_connection_id")
        chat_id = chat.get("id")

        if chat_id is None or not business_connection_id:
            return None

        return BusinessMessage(
            chat_id=str(chat_id),
            message_id=message.get("message_id"),
            business_connection_id=business_connection_id,
            from_user_id=user.get("id"),
            username=user.get("username"),
            first_name=user.get("first_name"),
            text=(message.get("text") or message.get("caption") or "").strip(),
            is_sender_business_bot=bool(message.get("sender_business_bot")),
            raw=message,
        )

    @staticmethod
    def parse_direct_message(update: dict[str, Any]) -> dict[str, Any] | None:
        message = update.get("message")
        return message if isinstance(message, dict) else None
