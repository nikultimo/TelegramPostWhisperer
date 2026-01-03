"""
Settings module for loading configuration from environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла (если существует)
# Ищем .env файл в корне проекта (на уровень выше backend/)
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(env_path, override=False)  # override=False: системные переменные имеют приоритет
else:
    # Также пробуем загрузить из текущей директории (для совместимости)
    load_dotenv(override=False)


# OpenRouter API settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")  # Support both for backward compatibility
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# OpenRouter attribution headers
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "postsender.sevostianovs.ru")
OPENROUTER_X_TITLE = os.getenv("OPENROUTER_X_TITLE", "PostSenderTG")


def get_openrouter_headers() -> dict:
    """
    Get OpenRouter attribution headers for API requests.
    
    Returns:
        Dictionary with HTTP-Referer and X-Title headers if configured.
    """
    headers = {}
    
    if OPENROUTER_HTTP_REFERER:
        # Ensure HTTP-Referer includes protocol if not present
        referer = OPENROUTER_HTTP_REFERER
        if not referer.startswith(("http://", "https://")):
            referer = f"https://{referer}"
        headers["HTTP-Referer"] = referer
    
    if OPENROUTER_X_TITLE:
        headers["X-Title"] = OPENROUTER_X_TITLE
    
    return headers

