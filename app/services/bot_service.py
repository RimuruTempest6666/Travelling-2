from __future__ import annotations

import time
from typing import Any

from app.ai.gemini import GeminiClient
from app.config import Settings
from app.database import BotDatabase
from app.models import AiDecision, BusinessMessage
from app.services.safety import SafetyClassifier
from app.telegram.client import TelegramClient
from app.telegram.update_parser import TelegramUpdateParser


class BotService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: BotDatabase,
        telegram: TelegramClient,
        gemini: GeminiClient,
        safety: SafetyClassifier,
    ):
        self.settings = settings
        self.db = database
        self.telegram = telegram
        self.gemini = gemini
        self.safety = safety
        self.parser = TelegramUpdateParser()

    async def handle_update(self, update: dict[str, Any]) -> None:
        update_type = self.parser.get_update_type(update)

        if update_type == "business_connection":
            self.db.save_business_connection(update["business_connection"])
            return

        if update_type == "business_message":
            message = self.parser.parse_business_message(update)
            if message:
                await self.handle_business_message(message)
            return

        if update_type == "message":
            message = self.parser.parse_direct_message(update)
            if message:
                await self.handle_direct_command(message)
            return

    async def handle_business_message(self, message: BusinessMessage) -> None:
        if message.is_sender_business_bot:
            return

        if self._is_echo_message(message):
            return

        business_owner_id = await self._get_business_owner_id(message.business_connection_id)
        if self._is_owner_message(message, business_owner_id):
            self._save_owner_message(message)
            return

        self._save_user_message(message)

        if self.db.is_chat_blocked(message.chat_id):
            return

        mode = self.db.get_setting("mode", self.settings.safe_bot_mode) or self.settings.safe_bot_mode
        if mode == "silent":
            return

        recent_rows = self.db.recent_rows(message.chat_id, self.settings.recent_messages_limit)
        recent_context = self.db.format_rows(recent_rows)
        relevant_memory = self.db.search_messages(message.chat_id, message.text, self.settings.search_messages_limit)
        user_info = self._format_user_info(message)

        try:
            decision = await self.gemini.decide(
                incoming_text=message.text or "[non-text message]",
                recent_context=recent_context,
                relevant_memory=relevant_memory,
                user_info=user_info,
                prompt_path=self.settings.prompt_path,
                knowledge_path=self.settings.knowledge_path,
            )
            self.db.set_setting("last_ai_error", "")
        except Exception as exc:
            self.db.set_setting("last_ai_error", str(exc)[:500])
            await self._notify_owner(
                f"⚠️ AI error\nchat_id: {message.chat_id}\nmessage: {message.text or '[non-text]'}\nerror: {str(exc)[:500]}"
            )
            return

        await self._process_decision(message, decision, mode)

    async def handle_direct_command(self, message: dict[str, Any]) -> None:
        text = (message.get("text") or "").strip()
        if not text.startswith("/"):
            return

        chat_id = message.get("chat", {}).get("id")
        from_user_id = message.get("from", {}).get("id")
        command = text.split()[0].split("@")[0].lower()
        args = text.split()[1:]

        if command == "/start":
            await self.telegram.send_message(
                chat_id,
                f"Привет! Твой Telegram user_id: {from_user_id}\nВставь его в Railway Variables как OWNER_TELEGRAM_ID.",
            )
            return

        if self.settings.owner_telegram_id and from_user_id != self.settings.owner_telegram_id:
            await self.telegram.send_message(chat_id, "Команды доступны только владельцу.")
            return

        if command == "/status":
            await self.telegram.send_message(chat_id, self._status_text())
            return

        if command == "/pause":
            self.db.set_setting("global_paused", "1")
            await self.telegram.send_message(chat_id, "Готово. Бот поставлен на паузу.")
            return

        if command == "/resume":
            self.db.set_setting("global_paused", "0")
            await self.telegram.send_message(chat_id, "Готово. Бот снова включён.")
            return

        if command == "/mode":
            if not args or args[0] not in {"auto", "draft", "silent"}:
                await self.telegram.send_message(chat_id, "Используй: /mode auto, /mode draft или /mode silent")
                return
            self.db.set_setting("mode", args[0])
            await self.telegram.send_message(chat_id, f"Готово. Режим: {args[0]}")
            return

        await self.telegram.send_message(chat_id, "Команды: /status, /pause, /resume, /mode auto, /mode draft, /mode silent")

    async def _process_decision(self, message: BusinessMessage, decision: AiDecision, mode: str) -> None:
        critical = decision.critical or self.safety.is_critical(message.text)

        if decision.action == "ignore":
            return

        if mode == "draft":
            await self._notify_owner(
                f"📝 Черновик ответа\n\nchat_id: {message.chat_id}\nСообщение: {message.text or '[non-text]'}\n\nОтвет:\n{decision.answer or '—'}\n\nconfidence: {decision.confidence:.2f}\ncritical: {critical}\nreason: {decision.reason or '—'}"
            )
            return

        if decision.action == "handoff" or critical:
            await self._notify_owner(
                f"🔴 Критичный handoff\n\nchat_id: {message.chat_id}\nСообщение: {message.text or '[non-text]'}\n\nЧерновик:\n{decision.answer or '—'}\n\nconfidence: {decision.confidence:.2f}\nreason: {decision.reason or '—'}"
            )
            if self.settings.send_handoff_to_chat and decision.answer:
                await self._send_business_answer(message, decision.answer)
            return

        if not decision.answer:
            await self._notify_owner(f"⚠️ Empty AI answer\nchat_id: {message.chat_id}\nmessage: {message.text or '[non-text]'}")
            return

        await self._send_business_answer(message, decision.answer)

    async def _send_business_answer(self, source_message: BusinessMessage, text: str) -> None:
        data = await self.telegram.send_message(
            source_message.chat_id,
            text,
            business_connection_id=source_message.business_connection_id,
        )
        message_id = None
        try:
            message_id = data.get("result", {}).get("message_id")
        except AttributeError:
            message_id = None

        self.db.mark_outgoing_ignored(source_message.chat_id, source_message.business_connection_id, message_id)
        self.db.save_message(
            chat_id=source_message.chat_id,
            business_connection_id=source_message.business_connection_id,
            telegram_message_id=message_id,
            from_user_id=None,
            sender_type="assistant",
            text=text,
        )

    async def _notify_owner(self, text: str) -> None:
        if self.settings.owner_telegram_id:
            await self.telegram.send_message(self.settings.owner_telegram_id, text)

    def _is_echo_message(self, message: BusinessMessage) -> bool:
        return self.db.is_ignored_outgoing(
            message.chat_id,
            message.business_connection_id,
            message.message_id,
        ) or self.db.is_recent_assistant_echo(message.chat_id, message.text)

    async def _get_business_owner_id(self, business_connection_id: str) -> int | None:
        cached = self.db.get_business_connection(business_connection_id)
        if cached and cached.get("user_id"):
            return int(cached["user_id"])

        try:
            connection = await self.telegram.get_business_connection(business_connection_id)
            if connection:
                self.db.save_business_connection(connection)
                user = connection.get("user") or {}
                return user.get("id")
        except Exception as exc:
            self.db.set_setting("last_telegram_error", str(exc)[:500])

        return None

    def _is_owner_message(self, message: BusinessMessage, business_owner_id: int | None) -> bool:
        if self.settings.owner_telegram_id and message.from_user_id == self.settings.owner_telegram_id:
            return True
        if business_owner_id and message.from_user_id == business_owner_id:
            return True
        return False

    def _save_owner_message(self, message: BusinessMessage) -> None:
        self.db.save_message(
            chat_id=message.chat_id,
            business_connection_id=message.business_connection_id,
            telegram_message_id=message.message_id,
            from_user_id=message.from_user_id,
            sender_type="owner",
            text=message.text or "[owner non-text message]",
        )
        self.db.set_manual_takeover(message.chat_id, self.settings.manual_takeover_minutes)

    def _save_user_message(self, message: BusinessMessage) -> None:
        self.db.save_message(
            chat_id=message.chat_id,
            business_connection_id=message.business_connection_id,
            telegram_message_id=message.message_id,
            from_user_id=message.from_user_id,
            sender_type="user",
            text=message.text or "[non-text message]",
        )

    @staticmethod
    def _format_user_info(message: BusinessMessage) -> str:
        return (
            f"chat_id={message.chat_id}; "
            f"user_id={message.from_user_id}; "
            f"username=@{message.username}; "
            f"first_name={message.first_name}"
        )

    def _status_text(self) -> str:
        return (
            f"mode: {self.db.get_setting('mode', self.settings.safe_bot_mode)}\n"
            f"paused: {self.db.get_setting('global_paused', '0')}\n"
            f"has_gemini_key: {bool(self.settings.gemini_api_key)}\n"
            f"gemini_model: {self.settings.gemini_model}\n"
            f"owner_id_set: {bool(self.settings.owner_telegram_id)}\n"
            f"confidence_threshold: {self.settings.confidence_threshold}\n"
            f"recent_messages_limit: {self.settings.recent_messages_limit}\n"
            f"last_ai_error: {self.db.get_setting('last_ai_error', '') or '—'}"
        )
