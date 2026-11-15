import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv(Path(__file__).resolve().with_name(".env"))

class Config:
    # Database URL for SQLite persistent storage
    DATABASE_URL = "sqlite:///market_data.db"

    # Trading Parameters for Option Selling Strategy
    SYMBOL = "NIFTY"
    EXCHANGE = "NFO"
    MIN_IV_PERCENTILE = 60.0  # Confluence Gatekeeper threshold
    PROFIT_TARGET_DECAY = 0.50  # Exit when 50% premium is decayed
    STOP_LOSS_MULTIPLIER = 1.5  # Exit if premium reaches 1.5x of credit received
    UPSTOX_OPTION_EXPIRY = os.getenv("UPSTOX_OPTION_EXPIRY", "current_week")
    OPTION_STRIKE_WINDOW = int(os.getenv("OPTION_STRIKE_WINDOW", "10"))
    OPTION_CHAIN_REFRESH_SECONDS = int(os.getenv("OPTION_CHAIN_REFRESH_SECONDS", "300"))
    OPTION_HISTORY_INTERVAL_SECONDS = float(os.getenv("OPTION_HISTORY_INTERVAL_SECONDS", "1"))
    OPTION_HISTORY_RETENTION_DAYS = int(os.getenv("OPTION_HISTORY_RETENTION_DAYS", "60"))
    MARKET_HISTORY_INTERVAL_SECONDS = float(os.getenv("MARKET_HISTORY_INTERVAL_SECONDS", "60"))
    FEED_STALE_AFTER_SECONDS = int(os.getenv("FEED_STALE_AFTER_SECONDS", "30"))
    POSITION_IV_SPIKE_PERCENT = float(os.getenv("POSITION_IV_SPIKE_PERCENT", "25"))
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))
    AGENT_LOOP_COOLDOWN_SECONDS = float(
        os.getenv("AGENT_LOOP_COOLDOWN_SECONDS", "60")
    )

    # Broker & Telegram API Placeholders (Update in .env file)
    BROKER_API_KEY = os.getenv("BROKER_API_KEY", "YOUR_BROKER_API_KEY")
    BROKER_API_SECRET = os.getenv("BROKER_API_SECRET", "YOUR_BROKER_API_SECRET")
    UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

config = Config()