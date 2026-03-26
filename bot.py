import asyncio
import os
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from groq import Groq

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Groq client ──────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

conversation_history: dict[int, list[dict]] = {}

SYSTEM_PROMPT = (
    "Ты полезный AI-ассистент. Отвечай чётко, по существу и на том языке, "
    "на котором пишет пользователь. Если не знаешь ответа — честно скажи об этом."
)

MAX_HISTORY = 20


# ── Keep-alive HTTP server ────────────────────────────────────────────────────
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_history(chat_id: int) -> list[dict]:
    return conversation_history.setdefault(chat_id, [])


def trim_history(chat_id: int) -> None:
    h = get_history(chat_id)
    if len(h) > MAX_HISTORY:
        conversation_history[chat_id] = h[-MAX_HISTORY:]


async def ask_groq(chat_id: int, user_text: str) -> str:
    history = get_history(chat_id)
    history.append({"role": "user", "content": user_text})
    trim_history(chat_id)

    response = groq_client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        temperature=0.7,
        max_tokens=1024,
    )

    reply = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply})
    return reply


# ── Handlers ─────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Привет! Я AI-ассистент на базе Groq.\n\n"
        "Просто напиши мне что-нибудь, и я отвечу.\n\n"
        "Команды:\n"
        "  /start — это сообщение\n"
        "  /clear — очистить историю диалога\n"
        "  /model — узнать текущую модель"
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conversation_history.pop(update.effective_chat.id, None)
    await update.message.reply_text("🗑 История диалога очищена.")


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    await update.message.reply_text(f"🤖 Текущая модель: `{model}`", parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        reply = await ask_groq(chat_id, user_text)
        await update.message.reply_text(reply)
    except Exception as exc:
        logger.exception("Groq error: %s", exc)
        await update.message.reply_text(
            "⚠️ Произошла ошибка при обращении к Groq. Попробуйте позже."
        )


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception: %s", context.error, exc_info=context.error)


# ── Bot (запускается в отдельном потоке) ──────────────────────────────────────
def run_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(handle_error)

    logger.info("Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


# ── Entry point — сначала HTTP, потом бот ────────────────────────────────────
def main() -> None:
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info("Health server listening on port %d", port)

    # Бот в отдельном потоке
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()

    # HTTP-сервер в главном потоке — Render видит открытый порт
    server.serve_forever()


if __name__ == "__main__":
    main()
