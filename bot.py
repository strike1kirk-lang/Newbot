import asyncio
import os
import logging
import base64
import json
import firebase_admin
from firebase_admin import credentials, firestore
from aiohttp import web
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from groq import Groq

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Firebase ──────────────────────────────────────────────────────────────────
cred_json = json.loads(os.environ["FIREBASE_CREDENTIALS"])
cred = credentials.Certificate(cred_json)
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── Groq ──────────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
chat_semaphores: dict[int, asyncio.Semaphore] = {}

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
SEARCH_MODEL = "compound-beta"
GROQ_TIMEOUT = 30
MAX_HISTORY = 10

SYSTEM_PROMPT = (
    "Ты полезный AI-ассистент. Отвечай КРАТКО и по делу — максимум 2-3 предложения "
    "если вопрос простой. Развёрнуто отвечай только если тебя явно просят объяснить "
    "подробно или вопрос сложный. Отвечай на том языке на котором пишет пользователь."
)

SUMMARY_PROMPT = (
    "Сделай краткое резюме этого разговора в 3-5 предложениях. "
    "Сохрани важные факты о пользователе, темы и договорённости. "
    "Отвечай на том же языке что и диалог."
)

SEARCH_DETECTION_PROMPT = (
    "Нужен ли веб-поиск для ответа на это сообщение? "
    "Ответь только 'да' или 'нет'. "
    "Поиск нужен если: вопрос про текущие события, погоду, цены, новости, "
    "актуальные данные или что-то что могло измениться недавно."
)

# ── Семафоры ──────────────────────────────────────────────────────────────────
def get_semaphore(chat_id: int) -> asyncio.Semaphore:
    if chat_id not in chat_semaphores:
        chat_semaphores[chat_id] = asyncio.Semaphore(1)
    return chat_semaphores[chat_id]

# ── Firebase helpers ──────────────────────────────────────────────────────────
def get_user_doc(chat_id: int):
    return db.collection("users").document(str(chat_id))

def load_user_data(chat_id: int) -> dict:
    doc = get_user_doc(chat_id).get()
    if doc.exists:
        return doc.to_dict()
    return {"history": [], "summary": "", "facts": ""}

def save_user_data(chat_id: int, data: dict):
    get_user_doc(chat_id).set(data)

# ── Groq helpers ──────────────────────────────────────────────────────────────
def groq_request(*args, **kwargs):
    return groq_client.chat.completions.create(*args, **kwargs)

async def needs_search(user_text: str) -> bool:
    """Быстро проверяет нужен ли поиск"""
    loop = asyncio.get_event_loop()
    try:
        response = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: groq_request(
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": SEARCH_DETECTION_PROMPT},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=5,
                temperature=0,
            )),
            timeout=10
        )
        answer = response.choices[0].message.content.strip().lower()
        return "да" in answer or "yes" in answer
    except Exception:
        return False

async def do_search(query: str) -> str:
    """Выполняет поиск через compound-beta"""
    loop = asyncio.get_event_loop()
    response = await asyncio.wait_for(
        loop.run_in_executor(None, lambda: groq_request(
            model=SEARCH_MODEL,
            messages=[{"role": "user", "content": query}],
            max_tokens=512,
        )),
        timeout=GROQ_TIMEOUT
    )
    return response.choices[0].message.content.strip()

async def summarize_history(history: list) -> str:
    loop = asyncio.get_event_loop()
    text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    response = await asyncio.wait_for(
        loop.run_in_executor(None, lambda: groq_request(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": text}
            ],
            max_tokens=256,
        )),
        timeout=GROQ_TIMEOUT
    )
    return response.choices[0].message.content.strip()

async def ask_groq(chat_id: int, user_text: str) -> str:
    loop = asyncio.get_event_loop()

    user_data = load_user_data(chat_id)
    history = user_data.get("history", [])
    summary = user_data.get("summary", "")
    facts = user_data.get("facts", "")

    system = SYSTEM_PROMPT
    if summary:
        system += f"\n\nРезюме прошлых разговоров:\n{summary}"
    if facts:
        system += f"\n\nФакты о пользователе:\n{facts}"

    # Проверяем нужен ли поиск
    search_result = None
    if await needs_search(user_text):
        logger.info("Search needed for: %s", user_text)
        search_result = await do_search(user_text)
        system += f"\n\nРезультат поиска:\n{search_result}"

    history.append({"role": "user", "content": user_text})

    response = await asyncio.wait_for(
        loop.run_in_executor(None, lambda: groq_request(
            model=TEXT_MODEL,
            messages=[{"role": "system", "content": system}] + history,
            temperature=0.7,
            max_tokens=512,
        )),
        timeout=GROQ_TIMEOUT
    )

    reply = (response.choices[0].message.content or "").strip()
    history.append({"role": "assistant", "content": reply})

    new_summary = summary
    if len(history) >= MAX_HISTORY:
        logger.info("Summarizing history for chat %d", chat_id)
        new_summary = await summarize_history(history)
        history = []

    user_data["history"] = history
    user_data["summary"] = new_summary
    save_user_data(chat_id, user_data)

    return reply

async def ask_groq_vision(chat_id: int, image_b64: str, caption: str) -> str:
    loop = asyncio.get_event_loop()
    user_data = load_user_data(chat_id)
    summary = user_data.get("summary", "")

    system = SYSTEM_PROMPT
    if summary:
        system += f"\n\nРезюме прошлых разговоров:\n{summary}"

    user_message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": caption if caption else "Что на этом фото?"}
        ]
    }

    response = await asyncio.wait_for(
        loop.run_in_executor(None, lambda: groq_request(
            model=VISION_MODEL,
            messages=[{"role": "system", "content": system}, user_message],
            temperature=0.7,
            max_tokens=512,
        )),
        timeout=GROQ_TIMEOUT
    )

    reply = response.choices[0].message.content.strip()

    history = user_data.get("history", [])
    history.append({"role": "user", "content": f"[фото] {caption}" if caption else "[фото]"})
    history.append({"role": "assistant", "content": reply})
    user_data["history"] = history
    save_user_data(chat_id, user_data)

    return reply

# ── Handlers ──────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Привет! Я AI-ассистент на базе Groq.\n\n"
        "Просто напиши мне что-нибудь 💬\n"
        "Ищу актуальную инфу когда нужно 🔍\n"
        "Вижу фото 📷\n"
        "Помню наши разговоры 🧠\n\n"
        "/clear — очистить память\n"
        "/memory — что помню о тебе"
    )

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    save_user_data(chat_id, {"history": [], "summary": "", "facts": ""})
    await update.message.reply_text("🗑 Память очищена.")

async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    data = load_user_data(chat_id)
    summary = data.get("summary", "")
    history_len = len(data.get("history", []))

    text = f"🧠 Память:\n\nСообщений в сессии: {history_len}\n\n"
    if summary:
        text += f"📖 Резюме:\n{summary}"
    else:
        text += "Пока пусто — начни общаться!"

    await update.message.reply_text(text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    async with get_semaphore(chat_id):
        try:
            reply = await ask_groq(chat_id, user_text)
            if len(reply) > 4096:
                reply = reply[:4090] + "..."
            await update.message.reply_text(reply)
        except asyncio.TimeoutError:
            await update.message.reply_text("⏱ Не ответил вовремя. Попробуй ещё раз.")
        except Exception as exc:
            logger.exception("Error: %s", exc)
            await update.message.reply_text("⚠️ Ошибка. Попробуй ещё раз.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    async with get_semaphore(chat_id):
        try:
            photo = update.message.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            photo_bytes = await file.download_as_bytearray()
            image_b64 = base64.b64encode(photo_bytes).decode("utf-8")
            caption = update.message.caption or ""
            reply = await ask_groq_vision(chat_id, image_b64, caption)
            if len(reply) > 4096:
                reply = reply[:4090] + "..."
            await update.message.reply_text(reply)
        except asyncio.TimeoutError:
            await update.message.reply_text("⏱ Не ответил вовремя. Попробуй ещё раз.")
        except Exception as exc:
            logger.exception("Photo error: %s", exc)
            await update.message.reply_text("⚠️ Не удалось обработать фото.")

async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception: %s", context.error, exc_info=context.error)

async def health(request):
    return web.Response(text="OK")

async def main() -> None:
    port = int(os.environ.get("PORT", 10000))
    app_web = web.Application()
    app_web.router.add_get("/", health)
    runner = web.AppRunner(app_web)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("Health server listening on port %d", port)

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_error_handler(handle_error)

    logger.info("Bot is running...")
    async with app:
        await app.start()
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
