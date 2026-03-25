# 🤖 Telegram Bot — Groq AI

Telegram-бот с AI на базе [Groq API](https://console.groq.com/), деплой на [Render](https://render.com/).

## Стек

| | |
|---|---|
| Язык | Python 3.11+ |
| Telegram | [python-telegram-bot v21](https://python-telegram-bot.org/) |
| AI | [Groq API](https://console.groq.com/) (LLaMA 3.3 70B по умолчанию) |
| Хостинг | [Render](https://render.com/) — Background Worker |

---

## 📁 Структура

```
telegram-groq-bot/
├── bot.py            # Основной код бота
├── requirements.txt  # Зависимости Python
├── .env.example      # Пример переменных окружения
├── .gitignore
└── README.md
```

---

## 🚀 Быстрый старт

### 1. Получи токены

**Telegram:**
1. Открой [@BotFather](https://t.me/BotFather) → `/newbot`
2. Скопируй `TELEGRAM_BOT_TOKEN`

**Groq:**
1. Зарегистрируйся на [console.groq.com](https://console.groq.com/)
2. Создай API-ключ → скопируй `GROQ_API_KEY`

---

### 2. Локальный запуск

```bash
# Клонируй репозиторий
git clone https://github.com/YOUR_USERNAME/telegram-groq-bot.git
cd telegram-groq-bot

# Создай виртуальное окружение
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Установи зависимости
pip install -r requirements.txt

# Создай .env
cp .env.example .env
# Открой .env и заполни токены

# Запусти бота
python bot.py
```

---

### 3. Деплой на Render

1. **Залей код на GitHub** (репозиторий может быть приватным)

2. **Создай сервис на Render:**
   - Зайди на [dashboard.render.com](https://dashboard.render.com/)
   - Нажми **New → Background Worker**
   - Подключи свой GitHub репозиторий

3. **Настройки сервиса:**

   | Параметр | Значение |
   |---|---|
   | **Environment** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `python bot.py` |

4. **Environment Variables** (вкладка Environment):

   | Ключ | Значение |
   |---|---|
   | `TELEGRAM_BOT_TOKEN` | твой токен от BotFather |
   | `GROQ_API_KEY` | твой ключ от Groq |
   | `GROQ_MODEL` | `llama-3.3-70b-versatile` (или другая) |

5. Нажми **Create Background Worker** — Render сам установит зависимости и запустит бота.

---

## 💬 Команды бота

| Команда | Описание |
|---|---|
| `/start` | Приветствие и список команд |
| `/clear` | Очистить историю диалога |
| `/model` | Показать текущую модель Groq |

---

## 🔧 Доступные модели Groq

Измени переменную `GROQ_MODEL` в Render:

| Модель | Описание |
|---|---|
| `llama-3.3-70b-versatile` | Лучшее качество (по умолчанию) |
| `llama-3.1-8b-instant` | Быстрее, легче |
| `mixtral-8x7b-32768` | Длинный контекст |
| `gemma2-9b-it` | Google Gemma 2 |

---

## ⚠️ Важные замечания

- История диалогов хранится **в памяти** — сбрасывается при перезапуске сервиса.
- На бесплатном тарифе Render сервис может "засыпать" — для бота это нормально, он Background Worker и не засыпает.
- Не коммить `.env` в git — используй переменные окружения Render.
