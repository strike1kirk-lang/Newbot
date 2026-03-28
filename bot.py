import asyncio
import os
import logging
import json
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

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
conversation_history: dict[int, list[dict]] = {}
chat_semaphores: dict[int, asyncio.Semaphore] = {}  # по одному на каждый чат

SYSTEM_PROMPT = (
    "Ты полезный AI-ассистент. Отвечай чётко, по существу и на том языке, "
    "на котором пишет пользователь. Если не знаешь ответа — честно скажи об этом. "
    "У тебя есть доступ к актуальной информации из интернета — используй его когда нужно."
)

MAX_HISTORY = 10
GROQ_TIMEOUT = 30

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}


def get_semaphore(chat_id: int) -> asyncio.Semaphore:
    if chat_id not in chat_semaphores:
        chat_semaphores[chat_id] = asyncio.Semaphore(1)
    return chat_semaphores[chat_id]


def get_history(chat_id: int) -> list[dict]:
    return conversation_history.setdefault(chat_id, [])


def trim_history(chat_id: int) -> None:
    h = get_history(chat_id)
    if len(h) > MAX_HISTORY:
        conversation_history[chat_id] = h[-MAX_HISTORY:]


def groq_request(*args, **kwargs):
    return groq_client.chat.completions.create(*args, **kwargs)


async def ask_groq(chat_id: int, user_text: str) -> str:
    history = get_history(chat_id)
    history.append({"role": "user", "content": user_text})
    trim_history(chat_id)
    loop = asyncio.get_event_loop()

    response = await asyncio.wait_for(
        loop.run_in_executor(None, lambda: groq_request(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            temperature=0.7,
            max_tokens=1024,
            tools=[WEB_SEARCH_TOOL],
            tool_choice="auto",
        )),
        timeout=GROQ_TIMEOUT
    )

    message = response.choices[0].message

    if message.tool_calls:
        history.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in message.tool_calls
            ]
        })

        for tool_call in message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            query = args.get("query", "")
            logger.info("Web search: %s", query)

            search_response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: groq_request(
                    model="compound-beta",
                    messages=[{"role": "user", "content": query}],
                    max_tokens=512,
                )),
                timeout=GROQ_TIMEOUT
            )
            search_result = search_response.choices[0].message.content
            history.append({"role": "tool", "tool_call_id": tool_call.id, "content": search_result})

        final_response = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: groq_request(
                model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                temperature=0.7,
                max_tokens=1024,
            )),
            timeout=GROQ_TIMEOUT
        )
        reply = final_response.choices[0].message.content.strip()
    else:
        reply = message.content.strip()

    history.append({"role": "assistant", "content": reply})
    return reply


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

    # Каждый чат обрабатывается по очереди, разные чаты — параллельно
    async with get_semaphore(chat_id):
        try:
            reply = await ask_groq(chat_id, user_text)
            await update.message.reply_text(reply)
        except asyncio.TimeoutError:
            logger.error("Groq timeout for chat %d", chat_id)
            await update.message.reply_text("⏱ Groq не ответил вовремя. Попробуй ещё раз.")
        except Exception as exc:
            logger.exception("Groq error: %s", exc)
            await update.message.reply_text("⚠️ Произошла ошибка. Попробуй ещё раз.")


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
