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

SYSTEM_PROMPT = (
    "Ты полезный AI-ассистент. Отвечай чётко, по существу и на том языке, "
    "на котором пишет пользователь. Если не знаешь ответа — честно скажи об этом. "
    "У тебя есть доступ к актуальной информации из интернета — используй его когда нужно."
)

SUMMARY_PROMPT = (
    "Ты помощник по сжатию диалогов. Сделай краткое резюме этого разговора "
    "в 3-5 предложениях. Сохрани самое важное: факты о пользователе, темы разговора, "
    "договорённости. Отвечай на том же языке что и диалог."
)

MAX_HISTORY = 10       # сообщений до сжатия
GROQ_TIMEOUT = 30
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

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

def load_history(chat_id: int) -> list:
    return load_user_data(chat_id).get("history", [])

def save_history(chat_id: int, history: list, summary: str = None, facts: str = None):
    data = load_user_data(chat_id)
    data["history"] = history
    if summary is not None:
        data["summary"] = summary
    if facts is not None:
        data["facts"] = facts
    save_user_data(chat_id, data)

# ── Groq helpers ──────────────────────────────────────────────────────────────
def groq_request(*args, **kwargs):
    return groq_client.chat.completions.create(*args, **kwargs)

async def summarize_history(history: list) -> str:
    """Сжимает историю в краткое резюме"""
    loop = asyncio.get_event_loop()
    text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    response = await asyncio.wait_for(
        loop.run_in_executor(None, lambda: groq_request(
            model="compound-beta",
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": text}
            ],
            max_tokens=512,
        )),
        timeout=GROQ_TIMEOUT
    )
    return response.choices[0].message.content.strip()

async def ask_groq(chat_id: int, user_text: str) -> str:
    loop = asyncio.get_event_loop()

    # Загружаем данные из Firebase
    user_data = load_user_data(chat_id)
    history = user_data.get("history", [])
    summary = user_data.get("summary", "")
    facts = user_data.get("facts", "")

    # Строим системный промпт с памятью
    system = SYSTEM_PROMPT
    if summary:
        system += f"\n\nКраткое резюме прошлых разговоров:\n{summary}"
    if facts:
        system += f"\n\nИзвестные факты о пользователе:\n{facts}"

    history.append({"role": "user", "content": user_text})

    response = await asyncio.wait_for(
        loop.run_in_executor(None, lambda: groq_request(
            model="compound-beta",
            messages=[{"role": "system", "content": system}] + history,
            temperature=0.7,
            max_tokens=1024,
        )),
        timeout=GROQ_TIMEOUT
    )

    reply = (response.choices[0].message.content or "").strip()
    history.append({"role": "assistant", "content": reply})

    # Если история длинная — сжимаем
    new_summary = summary
    if len(history) >= MAX_HISTORY:
        logger.info("Summarizing history for chat %d", chat_id)
        new_summary = await summarize_history(history)
        history = []  # очищаем после сжатия

    save_history(chat_id, history, summary=new_summary)
    return reply

async def ask_groq_vision(chat_id: int, image_b64: str, caption: str) -> str:
    loop = asyncio.get_event_loop()
    user_data = load_user_data(chat_id)
    summary = user_data.get("summary", "")
    facts = user_data.get("facts", "")

    system = SYSTEM_PROMPT
    if summary:
        system += f"\n\nКраткое резюме прошлых разговоров:\n{summary}"
    if facts:
        system += f"\n\nИзвестные факты о пользователе:\n{facts}"

    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            },
            {
                "type": "text",
                "text": caption if caption else "Что на этом фото?"
            }
        ]
    }

    response = await asyncio.wait_for(
        loop.run_in_executor(None, lambda: groq_request(
            model=VISION_MODEL,
            messages=[{"role": "system", "content": system}, user_message],
            temperature=0.7,
            max_tokens=1024,
        )),
        timeout=GROQ_TIMEOUT
    )

    reply = response.choices[0].message.content.strip()

    # Сохраняем фото в историю как текст
    history = user_data.get("history", [])
    history.append({"role": "user", "content": f"[фото] {caption}" if caption else "[фото]"})
    history.append({"role": "assistant", "content": reply})
    save_history(chat_id, history)

    return reply

# ── Handlers ──────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Привет! Я AI-ассистент на базе Groq.\n\n"
        "Просто напиши мне что-нибудь, и я отвечу.\n"
        "Могу искать актуальную информацию в интернете 🔍\n"
        "Могу смотреть на фотографии 📷\n"
        "Помню наши прошлые разговоры 🧠\n\n"
        "Команды:\n"
        "  /start — это сообщение\n"
        "  /clear — очистить историю\n"
        "  /memory — показать что помню о тебе\n"
        "  /model — текущие модели"
    )

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    save_user_data(chat_id, {"history": [], "summary": "", "facts": ""})
    await update.message.reply_text("🗑 История и память очищены.")

async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    data = load_user_data(chat_id)
    summary = data.get("summary", "")
    facts = data.get("facts", "")
    history_len = len(data.get("history", []))

    text = f"🧠 Моя память о тебе:\n\n"
    text += f"📝 Сообщений в текущей сессии: {history_len}\n\n"
    if summary:
        text += f"📖 Резюме прошлых разговоров:\n{summary}\n\n"
    if facts:
        text += f"👤 Факты о тебе:\n{facts}"
    if not summary and not facts:
        text += "Пока ничего не помню — начни общаться!"

    await update.message.reply_text(text)

async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"🤖 Текстовая модель: `compound-beta`\n"
        f"👁 Vision модель: `{VISION_MODEL}`",
        parse_mode="Markdown"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    async with get_semaphore(chat_id):
        try:
            reply = await ask_groq(chat_id, user_text)
            # Telegram лимит 4096 символов
            if len(reply) > 4096:
                reply = reply[:4090] + "..."
            await update.message.reply_text(reply)
        except asyncio.TimeoutError:
            await update.message.reply_text("⏱ Groq не ответил вовремя. Попробуй ещё раз.")
        except Exception as exc:
            logger.exception("Groq error: %s", exc)
            await update.message.reply_text("⚠️ Произошла ошибка. Попробуй ещё раз.")

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
            await update.message.reply_text("⏱ Groq не ответил вовремя. Попробуй ещё раз.")
        except Exception as exc:
            logger.exception("Photo error: %s", exc)
            await update.message.reply_text("⚠️ Не удалось обработать фото. Попробуй ещё раз.")

async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception: %s", context.error, exc_info=context.error)

async def health(request):
    return web.Response(text="OK")

# ── Main ──────────────────────────────────────────────────────────────────────
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
    app.add_handler(CommandHandler("model", cmd_model))
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
