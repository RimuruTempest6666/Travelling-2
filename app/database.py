from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any


class BotDatabase:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def init_schema(self, default_mode: str) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS business_connections (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    can_reply INTEGER NOT NULL DEFAULT 0,
                    is_enabled INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_state (
                    chat_id TEXT PRIMARY KEY,
                    paused INTEGER NOT NULL DEFAULT 0,
                    takeover_until REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    business_connection_id TEXT,
                    telegram_message_id INTEGER,
                    from_user_id INTEGER,
                    sender_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ignored_outgoing_messages (
                    chat_id TEXT NOT NULL,
                    business_connection_id TEXT,
                    telegram_message_id INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(chat_id, business_connection_id, telegram_message_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_chat_created
                ON messages(chat_id, created_at DESC);
                """
            )

            try:
                conn.executescript(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                    USING fts5(text, content='messages', content_rowid='id');

                    CREATE TRIGGER IF NOT EXISTS messages_ai
                    AFTER INSERT ON messages BEGIN
                        INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
                    END;
                    """
                )
            except sqlite3.OperationalError:
                pass

            if self.get_setting("mode") is None:
                self.set_setting("mode", default_mode)

            conn.execute("DELETE FROM ignored_outgoing_messages WHERE created_at < ?", (time.time() - 86400,))
            conn.commit()

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            conn.commit()

    def save_business_connection(self, connection: dict[str, Any]) -> None:
        user = connection.get("user") or {}
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO business_connections(id, user_id, can_reply, is_enabled, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    user_id=excluded.user_id,
                    can_reply=excluded.can_reply,
                    is_enabled=excluded.is_enabled,
                    updated_at=excluded.updated_at
                """,
                (
                    connection.get("id"),
                    user.get("id"),
                    1 if connection.get("can_reply") else 0,
                    1 if connection.get("is_enabled") else 0,
                    time.time(),
                ),
            )
            conn.commit()

    def get_business_connection(self, connection_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM business_connections WHERE id=?", (connection_id,)).fetchone()
            return dict(row) if row else None

    def save_message(
        self,
        *,
        chat_id: str,
        business_connection_id: str | None,
        telegram_message_id: int | None,
        from_user_id: int | None,
        sender_type: str,
        text: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO messages(chat_id, business_connection_id, telegram_message_id, from_user_id, sender_type, text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (chat_id, business_connection_id, telegram_message_id, from_user_id, sender_type, text or "[empty]", time.time()),
            )
            conn.commit()

    def mark_outgoing_ignored(self, chat_id: str, business_connection_id: str | None, message_id: int | None) -> None:
        if not message_id:
            return
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO ignored_outgoing_messages(chat_id, business_connection_id, telegram_message_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, business_connection_id, message_id, time.time()),
            )
            conn.commit()

    def is_ignored_outgoing(self, chat_id: str, business_connection_id: str | None, message_id: int | None) -> bool:
        if not message_id:
            return False
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM ignored_outgoing_messages
                WHERE chat_id=? AND business_connection_id IS ? AND telegram_message_id=?
                """,
                (chat_id, business_connection_id, message_id),
            ).fetchone()
            return bool(row)

    def is_recent_assistant_echo(self, chat_id: str, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM messages
                WHERE chat_id=? AND sender_type='assistant' AND text=? AND created_at > ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (chat_id, text, time.time() - 120),
            ).fetchone()
            return bool(row)

    def set_manual_takeover(self, chat_id: str, minutes: int) -> None:
        until = time.time() + minutes * 60
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO chat_state(chat_id, paused, takeover_until) VALUES(?, 0, ?) ON CONFLICT(chat_id) DO UPDATE SET takeover_until=excluded.takeover_until",
                (chat_id, until),
            )
            conn.commit()

    def is_chat_blocked(self, chat_id: str) -> bool:
        with self.connect() as conn:
            if self.get_setting("global_paused", "0") == "1":
                return True
            row = conn.execute("SELECT paused, takeover_until FROM chat_state WHERE chat_id=?", (chat_id,)).fetchone()
            return bool(row and (row["paused"] or row["takeover_until"] > time.time()))

    def recent_rows(self, chat_id: str, limit: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT sender_type, text, created_at FROM messages WHERE chat_id=? ORDER BY created_at DESC LIMIT ?",
                (chat_id, limit),
            ).fetchall()
            return list(reversed(rows))

    @staticmethod
    def format_rows(rows: list[sqlite3.Row]) -> str:
        if not rows:
            return "Нет недавней истории."
        return "\n".join(f"{r['sender_type']}: {r['text']}" for r in rows)

    def search_messages(self, chat_id: str, query: str, limit: int) -> str:
        terms = re.findall(r"[\wа-яА-ЯёЁіІўЎ]{3,}", (query or "").lower(), flags=re.UNICODE)[:8]
        if not terms:
            return "Ничего не найдено."

        fts_query = " OR ".join(t + "*" for t in terms)
        with self.connect() as conn:
            try:
                rows = conn.execute(
                    """
                    SELECT m.sender_type, m.text, m.created_at
                    FROM messages_fts f
                    JOIN messages m ON m.id = f.rowid
                    WHERE messages_fts MATCH ? AND m.chat_id=?
                    ORDER BY bm25(messages_fts)
                    LIMIT ?
                    """,
                    (fts_query, chat_id, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(
                    "SELECT sender_type, text, created_at FROM messages WHERE chat_id=? AND lower(text) LIKE ? ORDER BY created_at DESC LIMIT ?",
                    (chat_id, f"%{terms[0]}%", limit),
                ).fetchall()

        return self.format_rows(list(rows)) if rows else "Ничего не найдено."
