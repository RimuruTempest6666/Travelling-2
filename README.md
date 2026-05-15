# Telegram Business AI Bot

Готовый минимальный проект Telegram Business AI-бота на **Python + FastAPI + Gemini + SQLite**.

## Что умеет

- принимает `business_message` от Telegram Business;
- отвечает через `business_connection_id`;
- режимы `auto`, `draft`, `silent`;
- поведение берёт из `data/PROMPT.md`;
- базу знаний берёт из `data/KNOWLEDGE.md`;
- хранит историю в SQLite;
- ищет старый контекст по конкретному чату;
- сложные вопросы передаёт владельцу;
- команды: `/pause`, `/resume`, `/mode auto`, `/mode draft`, `/mode silent`, `/status`.

## Куда вставлять API-ключи

Никуда в код их вставлять не нужно. На Railway они добавляются в **Variables**.

Главные переменные:

```env
TELEGRAM_BOT_TOKEN=сюда_токен_бота_из_BotFather
GEMINI_API_KEY=сюда_ключ_Gemini_из_Google_AI_Studio
OWNER_TELEGRAM_ID=сюда_твой_telegram_user_id
WEBHOOK_SECRET_TOKEN=любая_длинная_случайная_строка
BOT_MODE=draft
DATABASE_PATH=/app/data/bot.sqlite3
PROMPT_PATH=data/PROMPT.md
KNOWLEDGE_PATH=data/KNOWLEDGE.md
```

Важно: не вставляй реальные ключи в GitHub. Только в Railway Variables.

## Как узнать OWNER_TELEGRAM_ID

1. Задеплой бота или запусти локально.
2. Напиши своему боту `/start`.
3. Он покажет твой Telegram user id.
4. Вставь это число в `OWNER_TELEGRAM_ID`.

## Деплой на Railway

1. Открой Railway.
2. New Project → Deploy from GitHub repo.
3. Выбери этот репозиторий.
4. В Variables добавь переменные из примера выше.
5. Добавь Volume для SQLite:
   - Mount path: `/app/data`
   - DATABASE_PATH: `/app/data/bot.sqlite3`
6. В Settings → Networking сгенерируй публичный домен.
7. Скопируй домен в переменную `APP_BASE_URL`, например:

```env
APP_BASE_URL=https://your-app.up.railway.app
```

## Установка webhook

После деплоя нужно установить webhook. Локально на компьютере:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Заполни в `.env`:

```env
TELEGRAM_BOT_TOKEN=...
APP_BASE_URL=https://your-app.up.railway.app
WEBHOOK_SECRET_TOKEN=...
```

Потом:

```bash
python scripts/set_webhook.py
```

Если всё нормально, Telegram ответит `Webhook was set`.

## Telegram Business

В BotFather включи Business Mode:

```txt
/mybots → твой бот → Bot Settings → Business Mode → Turn on
```

Потом в Telegram:

```txt
Настройки → Telegram Business → Чат-боты → добавить бота
```

Лучше сначала разрешить боту отвечать только новым чатам, а не контактам.

## Режимы

`draft` — безопасный режим, бот присылает тебе черновики.

`auto` — бот отвечает клиентам сам.

`silent` — бот только сохраняет историю.

Переключение в Telegram:

```txt
/mode draft
/mode auto
/mode silent
```

## Как менять поведение

Открой `data/PROMPT.md` и измени правила поведения.

## Как менять факты

Открой `data/KNOWLEDGE.md` и вставь цены, услуги, график, ссылки, частые ответы.

## Локальный запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Проверка:

```txt
http://127.0.0.1:8000/
```
