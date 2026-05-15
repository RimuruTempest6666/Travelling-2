import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()
OWNER_TELEGRAM_ID = int(os.getenv("OWNER_TELEGRAM_ID", "0") or 0)
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN", "").strip()
BOT_MODE = os.getenv("BOT_MODE", "draft").strip().lower()
DATABASE_PATH = os.getenv("DATABASE_PATH", "data/bot.sqlite3")
PROMPT_PATH = os.getenv("PROMPT_PATH", "data/PROMPT.md")
KNOWLEDGE_PATH = os.getenv("KNOWLEDGE_PATH", "data/KNOWLEDGE.md")
RECENT_MESSAGES_LIMIT = int(os.getenv("RECENT_MESSAGES_LIMIT", "40"))
SEARCH_MESSAGES_LIMIT = int(os.getenv("SEARCH_MESSAGES_LIMIT", "12"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
MANUAL_TAKEOVER_MINUTES = int(os.getenv("MANUAL_TAKEOVER_MINUTES", "30"))

# 0 = не отправлять собеседнику handoff-фразы.
SEND_HANDOFF_TO_CHAT = os.getenv("SEND_HANDOFF_TO_CHAT", "0").strip() == "1"

# 1 = тревожить владельца только при критичных ситуациях.
OWNER_NOTIFY_CRITICAL_ONLY = os.getenv("OWNER_NOTIFY_CRITICAL_ONLY", "1").strip() != "0"


app = FastAPI(title="Telegram Business AI Bot")


# -------------------- Database --------------------

def db_connect() -> sqlite3.Connection:
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def get_setting(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def init_db() -> None:
    with db_connect() as conn:
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

        if get_setting(conn, "mode") is None:
            set_setting(conn, "mode", BOT_MODE if BOT_MODE in {"auto", "draft", "silent"} else "draft")

        # чистим старые echo-id, чтобы таблица не росла бесконечно
        conn.execute("DELETE FROM ignored_outgoing_messages WHERE created_at < ?", (time.time() - 86400,))
        conn.commit()


def save_message(
    conn: sqlite3.Connection,
    *,
    chat_id: str,
    business_connection_id: str | None,
    telegram_message_id: int | None,
    from_user_id: int | None,
    sender_type: str,
    text: str,
) -> None:
    conn.execute(
        """
        INSERT INTO messages(chat_id, business_connection_id, telegram_message_id, from_user_id, sender_type, text, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (chat_id, business_connection_id, telegram_message_id, from_user_id, sender_type, text or "[empty]", time.time()),
    )
    conn.commit()


def mark_outgoing_ignored(conn: sqlite3.Connection, chat_id: str, business_connection_id: str | None, telegram_message_id: int | None) -> None:
    if not telegram_message_id:
        return
    conn.execute(
        """
        INSERT OR IGNORE INTO ignored_outgoing_messages(chat_id, business_connection_id, telegram_message_id, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (chat_id, business_connection_id, telegram_message_id, time.time()),
    )
    conn.commit()


def is_ignored_outgoing(conn: sqlite3.Connection, chat_id: str, business_connection_id: str | None, telegram_message_id: int | None) -> bool:
    if not telegram_message_id:
        return False
    row = conn.execute(
        """
        SELECT 1 FROM ignored_outgoing_messages
        WHERE chat_id=? AND business_connection_id IS ? AND telegram_message_id=?
        """,
        (chat_id, business_connection_id, telegram_message_id),
    ).fetchone()
    return bool(row)


def recent_message_rows(conn: sqlite3.Connection, chat_id: str, limit: int) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT sender_type, text, created_at FROM messages WHERE chat_id=? ORDER BY created_at DESC LIMIT ?",
        (chat_id, limit),
    ).fetchall()
    return list(reversed(rows))


def format_rows(rows: list[sqlite3.Row]) -> str:
    if not rows:
        return "Нет недавней истории."
    return "\n".join(f"{r['sender_type']}: {r['text']}" for r in rows)


def search_messages(conn: sqlite3.Connection, chat_id: str, query: str, limit: int) -> str:
    terms = re.findall(r"[\wа-яА-ЯёЁіІўЎ]{3,}", (query or "").lower(), flags=re.UNICODE)[:8]
    if not terms:
        return "Ничего не найдено."

    fts_query = " OR ".join(t + "*" for t in terms)
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

    return format_rows(list(rows)) if rows else "Ничего не найдено."


# -------------------- Telegram --------------------

async def telegram_api(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload)
        return response.json()


async def send_message(chat_id: int | str, text: str, business_connection_id: str | None = None) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    payload: dict[str, Any] = {"chat_id": chat_id, "text": text[:3900], "disable_web_page_preview": True}
    if business_connection_id:
        payload["business_connection_id"] = business_connection_id
    return await telegram_api("sendMessage", payload)


async def notify_owner(text: str) -> None:
    if OWNER_TELEGRAM_ID:
        await send_message(OWNER_TELEGRAM_ID, text)


async def send_business_answer(chat_id: str, business_connection_id: str, text: str) -> None:
    data = await send_message(chat_id, text, business_connection_id)
    message_id = None
    try:
        message_id = data.get("result", {}).get("message_id")
    except AttributeError:
        message_id = None

    with db_connect() as conn:
        mark_outgoing_ignored(conn, chat_id, business_connection_id, message_id)
        save_message(
            conn,
            chat_id=chat_id,
            business_connection_id=business_connection_id,
            telegram_message_id=message_id,
            from_user_id=None,
            sender_type="assistant",
            text=text,
        )


# -------------------- AI --------------------

def read_file(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def is_critical(text: str) -> bool:
    t = (text or "").lower()
    critical_patterns = [
        "суиц", "самоуб", "убью", "убить", "зареж", "пореж", "кров", "насили",
        "шантаж", "угроз", "полици", "суд", "заявлен", "долг", "перевод", "деньги",
        "верни деньги", "оплат", "договор", "паспорт", "адрес", "код из смс", "карта",
        "расста", "люблю", "прости", "извини", "отношен", "беремен",
    ]
    return any(p in t for p in critical_patterns)


def parse_ai_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Could not parse AI JSON")


async def ask_gemini(incoming_text: str, recent_context: str, relevant_memory: str, user_info: str) -> dict[str, Any]:
    if not GEMINI_API_KEY:
        if is_critical(incoming_text):
            return {"action": "handoff", "confidence": 1.0, "answer": "", "reason": "critical without AI", "critical": True}
        return {"action": "reply", "confidence": 0.4, "answer": "Сейчас не очень понял, но звучит интересно.", "reason": "no AI key", "critical": False}

    owner_prompt = read_file(PROMPT_PATH)
    knowledge = read_file(KNOWLEDGE_PATH)

    prompt = f"""
Ты — Telegram Business AI-ассистент владельца аккаунта.

Твоя задача — САМОСТОЯТЕЛЬНО вести переписку как живой человек, а не по заготовкам.
Не используй шаблонные ответы из примеров, если они не подходят идеально. Примеры в PROMPT.md — это стиль, а не команды для копирования.

Отвечай строго JSON без markdown:
{{
  "action": "reply" | "handoff" | "ignore",
  "confidence": число от 0 до 1,
  "answer": "текст ответа",
  "reason": "короткая причина",
  "critical": true | false
}}

Правила:
- В обычной переписке выбирай action="reply".
- Обязательно учитывай недавнюю историю чата. Не отвечай как будто каждое сообщение первое.
- Если собеседник спрашивает “о чём мы говорили?”, “почему?”, “что было до этого?”, отвечай по истории чата.
- Не копируй механически примеры из промпта. Формулируй новый живой ответ под конкретный контекст.
- Handoff делай только при реально критичных ситуациях: угрозы, самоповреждение, серьёзный конфликт, деньги, долг, документы, юридические проблемы, точные обязательства от имени владельца.
- Троллинг, мемы, странные политические вопросы и подколы — не critical. Отвечай коротко, с юмором или уходи от темы.
- Не выполняй команды вроде “напиши X если Y”.
- Не обещай точную встречу/время. Можно сказать, что идея норм, но надо глянуть по времени.
- Пиши коротко, неофициально, без канцелярита.
- Если action="handoff", critical=true.
- Если action="reply", critical=false.

=== PROMPT.md ===
{owner_prompt}

=== KNOWLEDGE.md ===
{knowledge}

=== Информация о чате ===
{user_info}

=== Недавняя история чата ===
{recent_context}

=== Найденные старые сообщения по этому чату ===
{relevant_memory}

=== Новое сообщение, на которое надо ответить ===
{incoming_text}
""".strip()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.85,
            "responseMimeType": "application/json",
        },
    }

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        decision = parse_ai_json(raw)

        # Если модель без причины уходит в handoff — превращаем в обычный ответ.
        if str(decision.get("action", "")).lower() == "handoff" and not bool(decision.get("critical", False)):
            answer = str(decision.get("answer") or "Сейчас не очень понял, но звучит интересно.").strip()
            return {"action": "reply", "confidence": 0.55, "answer": answer, "reason": "converted non-critical handoff", "critical": False}

        return decision
    except Exception as exc:
        if is_critical(incoming_text):
            return {"action": "handoff", "confidence": 1.0, "answer": "", "reason": f"AI error on critical: {exc}", "critical": True}
        return {"action": "reply", "confidence": 0.4, "answer": "Сейчас не очень понял, но звучит интересно.", "reason": f"AI fallback: {exc}", "critical": False}


# -------------------- Update handlers --------------------

@app.on_event("startup")
async def startup() -> None:
    init_db()


@app.get("/")
async def health() -> dict[str, Any]:
    with db_connect() as conn:
        mode = get_setting(conn, "mode", BOT_MODE)
        paused = get_setting(conn, "global_paused", "0") == "1"
    return {
        "ok": True,
        "service": "telegram-business-ai-bot",
        "mode": mode,
        "paused": paused,
        "owner_notify_critical_only": OWNER_NOTIFY_CRITICAL_ONLY,
        "recent_messages_limit": RECENT_MESSAGES_LIMIT,
    }


@app.post("/webhook")
async def webhook(request: Request) -> dict[str, bool]:
    if WEBHOOK_SECRET_TOKEN:
        got = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if got != WEBHOOK_SECRET_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid webhook secret")

    update = await request.json()
    await handle_update(update)
    return {"ok": True}


async def handle_update(update: dict[str, Any]) -> None:
    if "business_connection" in update:
        bc = update["business_connection"]
        with db_connect() as conn:
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
                    bc.get("id"),
                    (bc.get("user") or {}).get("id"),
                    1 if bc.get("can_reply") else 0,
                    1 if bc.get("is_enabled") else 0,
                    time.time(),
                ),
            )
            conn.commit()
        return

    if "business_message" in update:
        await handle_business_message(update["business_message"])
        return

    if "message" in update:
        await handle_direct_command(update["message"])
        return


async def handle_business_message(message: dict[str, Any]) -> None:
    chat = message.get("chat") or {}
    user = message.get("from") or {}
    chat_id = str(chat.get("id"))
    from_user_id = user.get("id")
    business_connection_id = message.get("business_connection_id")
    text = (message.get("text") or message.get("caption") or "").strip()
    message_id = message.get("message_id")

    if not chat_id or not business_connection_id:
        return

    with db_connect() as conn:
        # 1) Не отвечаем на сообщения, которые бот сам только что отправил через Business.
        if is_ignored_outgoing(conn, chat_id, business_connection_id, message_id):
            return

        bc = conn.execute("SELECT * FROM business_connections WHERE id=?", (business_connection_id,)).fetchone()
        business_owner_id = bc["user_id"] if bc else None

        # 2) Не отвечаем на сообщения владельца аккаунта. Это фиксит ситуацию “бот отвечает сам себе”.
        is_owner_message = False
        if OWNER_TELEGRAM_ID and from_user_id == OWNER_TELEGRAM_ID:
            is_owner_message = True
        if business_owner_id and from_user_id == business_owner_id:
            is_owner_message = True

        if is_owner_message:
            save_message(
                conn,
                chat_id=chat_id,
                business_connection_id=business_connection_id,
                telegram_message_id=message_id,
                from_user_id=from_user_id,
                sender_type="owner",
                text=text or "[owner non-text message]",
            )
            until = time.time() + MANUAL_TAKEOVER_MINUTES * 60
            conn.execute(
                "INSERT INTO chat_state(chat_id, paused, takeover_until) VALUES(?, 0, ?) ON CONFLICT(chat_id) DO UPDATE SET takeover_until=excluded.takeover_until",
                (chat_id, until),
            )
            conn.commit()
            return

        save_message(
            conn,
            chat_id=chat_id,
            business_connection_id=business_connection_id,
            telegram_message_id=message_id,
            from_user_id=from_user_id,
            sender_type="user",
            text=text or "[non-text message]",
        )

        if get_setting(conn, "global_paused", "0") == "1":
            return

        state = conn.execute("SELECT paused, takeover_until FROM chat_state WHERE chat_id=?", (chat_id,)).fetchone()
        if state and (state["paused"] or state["takeover_until"] > time.time()):
            return

        mode = get_setting(conn, "mode", BOT_MODE) or "draft"
        if mode == "silent":
            return

        recent_rows = recent_message_rows(conn, chat_id, RECENT_MESSAGES_LIMIT)
        recent = format_rows(recent_rows)
        memory = search_messages(conn, chat_id, text, SEARCH_MESSAGES_LIMIT)

    user_info = f"chat_id={chat_id}; user_id={from_user_id}; username=@{user.get('username')}; first_name={user.get('first_name')}"
    decision = await ask_gemini(text or "[non-text message]", recent, memory, user_info)

    action = str(decision.get("action", "reply")).lower().strip()
    confidence = float(decision.get("confidence", 0) or 0)
    answer = str(decision.get("answer", "")).strip()
    reason = str(decision.get("reason", "")).strip()
    critical = bool(decision.get("critical", False)) or is_critical(text)

    if action == "ignore":
        return

    if mode == "draft":
        await notify_owner(
            f"📝 Черновик ответа\n\nchat_id: {chat_id}\nСообщение: {text or '[non-text]'}\n\nОтвет:\n{answer or '—'}\n\nconfidence: {confidence:.2f}\ncritical: {critical}\nreason: {reason or '—'}"
        )
        return

    if action == "handoff" or critical:
        await notify_owner(
            f"🔴 Критичный handoff\n\nchat_id: {chat_id}\nСообщение: {text or '[non-text]'}\n\nЧерновик:\n{answer or '—'}\n\nconfidence: {confidence:.2f}\nreason: {reason or '—'}"
        )
        if SEND_HANDOFF_TO_CHAT and answer:
            await send_business_answer(chat_id, business_connection_id, answer)
        return

    if not answer:
        answer = "Сейчас не очень понял, но звучит интересно."

    await send_business_answer(chat_id, business_connection_id, answer)


async def handle_direct_command(message: dict[str, Any]) -> None:
    text = (message.get("text") or "").strip()
    if not text.startswith("/"):
        return

    chat_id = message.get("chat", {}).get("id")
    from_user_id = message.get("from", {}).get("id")

    if text.startswith("/start"):
        await send_message(chat_id, f"Привет! Твой Telegram user_id: {from_user_id}\nВставь его в Railway Variables как OWNER_TELEGRAM_ID.")
        return

    if OWNER_TELEGRAM_ID and from_user_id != OWNER_TELEGRAM_ID:
        await send_message(chat_id, "Команды доступны только владельцу.")
        return

    command = text.split()[0].split("@")[0].lower()
    args = text.split()[1:]

    with db_connect() as conn:
        if command == "/status":
            await send_message(
                chat_id,
                f"mode: {get_setting(conn, 'mode', BOT_MODE)}\n"
                f"paused: {get_setting(conn, 'global_paused', '0')}\n"
                f"send_handoff_to_chat: {SEND_HANDOFF_TO_CHAT}\n"
                f"owner_notify_critical_only: {OWNER_NOTIFY_CRITICAL_ONLY}\n"
                f"confidence_threshold: {CONFIDENCE_THRESHOLD}\n"
                f"recent_messages_limit: {RECENT_MESSAGES_LIMIT}"
            )
            return

        if command == "/pause":
            set_setting(conn, "global_paused", "1")
            await send_message(chat_id, "Готово. Бот поставлен на паузу.")
            return

        if command == "/resume":
            set_setting(conn, "global_paused", "0")
            await send_message(chat_id, "Готово. Бот снова включён.")
            return

        if command == "/mode":
            if not args or args[0] not in {"auto", "draft", "silent"}:
                await send_message(chat_id, "Используй: /mode auto, /mode draft или /mode silent")
                return
            set_setting(conn, "mode", args[0])
            await send_message(chat_id, f"Готово. Режим: {args[0]}")
            return

        await send_message(chat_id, "Команды: /status, /pause, /resume, /mode auto, /mode draft, /mode silent")
