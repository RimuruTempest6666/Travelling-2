from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BusinessMessage:
    chat_id: str
    message_id: int | None
    business_connection_id: str
    from_user_id: int | None
    username: str | None
    first_name: str | None
    text: str
    is_sender_business_bot: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class AiDecision:
    action: str
    confidence: float
    answer: str
    reason: str
    critical: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AiDecision":
        action = str(data.get("action", "reply")).lower().strip()
        if action not in {"reply", "handoff", "ignore"}:
            action = "reply"

        try:
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        return cls(
            action=action,
            confidence=max(0.0, min(1.0, confidence)),
            answer=str(data.get("answer") or "").strip(),
            reason=str(data.get("reason") or "").strip(),
            critical=bool(data.get("critical", False)),
        )
