import asyncio
import os
import logging
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
    "на котором пишет пользователь. Если не знаешь ответа — честно скажи об этом. "
    "У тебя есть доступ к актуальной информации из интернета — используй его когда нужно."
)

MAX_HISTORY = 20

# ── Web search tool ───────────────────────────────────────────────────────────
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    }
}


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
        tools=[WEB_SEARCH_TOOL],
        tool_choice="auto",
    )

    message = response.choices[0].message

    # Если модель решила сделать поиск
    if message.tool_calls:
        import json
        from groq import Groq as GroqClient

        # Добавляем ответ ассистента с tool_calls в историю
        history.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        })

        # Выполняем поиск через Groq compound-beta
        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            query = args.get("query", "")
            logger.info("Web search: %s", query)

            # Делаем отдельный запрос с compound-beta для реального поиска
            search_response = groq_client.chat.completions.create(
                model="compound-beta",
                messages=[{"role": "user", "content": query}],
                max_tokens=512,
            )
            search_result = search_response.choices[0].message.content

            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": search_result,
            })

        # Финальный ответ с результатами поиска
        final_response = groq_client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            temperature=0.7,
            max_tokens=1024,
        )
        reply = final_response.choices[0].message.content.strip()
    else:
        reply = message.content.strip()

    history.append({"role": "assistant", "content": reply})
    return reply


# ── Handlers ─────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Привет! Я AI-ассистент на базе Groq.\n\n"
        "Просто напиши мне что-нибудь, и я отвечу.\n"
        "Могу искать актуальную информацию в интернете 🔍\n\n"
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


# ── Health check endpoint ─────────────────────────────────────────────────────
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
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(handle_error)

    logger.info("Bot is running...")

    async with app:
        await app.start()
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
