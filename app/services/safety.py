from __future__ import annotations


class SafetyClassifier:
    CRITICAL_PATTERNS = [
        "суиц", "самоуб", "убью", "убить", "зареж", "пореж", "кров", "насили",
        "шантаж", "угроз", "полици", "суд", "заявлен", "долг", "перевод", "деньги",
        "верни деньги", "оплат", "договор", "паспорт", "код из смс", "карта",
        "расста", "люблю", "прости", "извини", "отношен", "беремен",
    ]

    def is_critical(self, text: str) -> bool:
        text = (text or "").lower()
        return any(pattern in text for pattern in self.CRITICAL_PATTERNS)
