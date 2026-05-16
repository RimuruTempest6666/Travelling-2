from __future__ import annotations

from typing import Any

import httpx


class TelegramClient:
    def __init__(self, token: str):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}" if token else ""

    async def request(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is empty")

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{self.base_url}/{method}", json=payload or {})
            return response.json()

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        business_connection_id: str | None = None,
    ) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text[:3900],
            "disable_web_page_preview": True,
        }
        if business_connection_id:
            payload["business_connection_id"] = business_connection_id

        return await self.request("sendMessage", payload)

    async def get_business_connection(self, business_connection_id: str) -> dict[str, Any] | None:
        data = await self.request("getBusinessConnection", {"business_connection_id": business_connection_id})
        if not data.get("ok"):
            return None
        return data.get("result")
