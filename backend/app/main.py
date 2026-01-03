from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import aiofiles
import yaml
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from openai import OpenAI

# Загружаем переменные окружения из .env файла (если существует)
# Ищем .env файл в корне проекта (на уровень выше backend/)
# Системные переменные окружения имеют приоритет над .env файлом
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(env_path, override=False)  # override=False: системные переменные имеют приоритет
else:
    # Также пробуем загрузить из текущей директории (для совместимости)
    load_dotenv(override=False)

from .telegram_sender import (
    InlineButton,
    TelegramBroadcastConfig,
    TelegramPhoto,
    TelegramVideo,
    TelegramSender,
    load_chat_ids_from_csv,
    save_blocked_users,
    load_blocked_users,
    save_blocked_users_async,
    load_blocked_users_async,
    is_blocked_user_error,
)
from .settings import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    get_openrouter_headers,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Telegram Broadcast API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

if (FRONTEND_DIR / "assets").exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIR / "assets"),
        name="assets",
    )


class InlineButtonPayload(BaseModel):
    text: str
    url: Optional[str] = None
    callback_data: Optional[str] = None


class PreviewPayload(BaseModel):
    message: str
    parse_mode: str = "Markdown"
    inline_keyboard: Optional[List[List[InlineButtonPayload]]] = None
    disable_web_page_preview: bool = False
    attach_message_to_first_photo: bool = False


class EnhancePayload(BaseModel):
    message: str
    parse_mode: str = "Markdown"


class BotInfo(BaseModel):
    name: str
    token: str


def load_bots_from_yaml() -> List[BotInfo]:
    """
    Load bot configurations from bots.yaml file.
    Returns empty list if file doesn't exist or is invalid.
    """
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).resolve().parents[2] / "bots.yaml",  # Project root
        Path("/app/bots.yaml"),  # Docker container path
        Path("bots.yaml"),  # Current working directory
    ]
    
    bots_yaml_path = None
    for path in possible_paths:
        if path.exists():
            bots_yaml_path = path
            logger.info(f"Found bots.yaml at: {bots_yaml_path}")
            break
    
    if not bots_yaml_path:
        logger.warning(f"bots.yaml file not found. Checked paths: {[str(p) for p in possible_paths]}")
        return []
    
    try:
        with open(bots_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        if not data or "bots" not in data:
            logger.warning("bots.yaml exists but has no 'bots' key")
            return []
        
        bots = []
        for bot_data in data["bots"]:
            if isinstance(bot_data, dict) and "name" in bot_data and "token" in bot_data:
                bots.append(BotInfo(name=bot_data["name"], token=bot_data["token"]))
        
        logger.info(f"Loaded {len(bots)} bot(s) from bots.yaml")
        return bots
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse bots.yaml: {e}")
        return []
    except Exception as e:
        logger.error(f"Error loading bots.yaml: {e}")
        return []


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    return {"status": "ok"}


@app.get("/api/bots", tags=["system"])
async def get_bots() -> dict:
    """
    Get list of available bots from bots.yaml file.
    """
    bots = await run_in_threadpool(load_bots_from_yaml)
    return {
        "bots": [{"name": bot.name, "token": bot.token} for bot in bots],
        "exists": len(bots) > 0
    }


@app.get("/api/csv-files", tags=["system"])
async def get_csv_files() -> dict:
    """
    Get list of available CSV files from data directory.
    """
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).resolve().parents[2] / "data",  # Project root/data
        Path("/app/data"),  # Docker container path
        Path("data"),  # Current working directory
    ]
    
    data_dir = None
    for path in possible_paths:
        if path.exists() and path.is_dir():
            data_dir = path
            logger.info(f"Found data directory at: {data_dir}")
            break
    
    if not data_dir:
        logger.warning(f"Data directory not found. Checked paths: {[str(p) for p in possible_paths]}")
        return {"files": [], "exists": False}
    
    csv_files = []
    try:
        # Collect all CSV files first
        csv_paths = list(data_dir.glob("*.csv"))
        
        # Process files asynchronously
        async def count_users(file_path: Path) -> int:
            try:
                async with aiofiles.open(file_path, "rb") as f:
                    csv_bytes = await f.read()
                    return len(await run_in_threadpool(load_chat_ids_from_csv, csv_bytes))
            except Exception as e:
                logger.warning(f"Failed to count users in {file_path.name}: {e}")
                return 0
        
        # Process all files concurrently
        user_counts = await asyncio.gather(*[count_users(fp) for fp in csv_paths])
        
        for file_path, user_count in zip(csv_paths, user_counts):
            csv_files.append({
                "name": file_path.name,
                "path": str(file_path),
                "user_count": user_count,
            })
        csv_files.sort(key=lambda x: x["name"])
        logger.info(f"Found {len(csv_files)} CSV file(s) in data directory")
    except Exception as e:
        logger.error(f"Error reading data directory: {e}")
        return {"files": [], "exists": False}
    
    return {
        "files": csv_files,
        "exists": len(csv_files) > 0
    }


@app.post("/api/preview", tags=["messaging"])
async def generate_preview(payload: PreviewPayload) -> dict:
    """
    Echo endpoint that simply returns the payload and allows the frontend
    to render a Telegram-style preview.
    """
    return payload.dict()


@app.post("/api/enhance", tags=["messaging"])
async def enhance_message(payload: EnhancePayload) -> dict:
    """
    Enhance message text using OpenRouter AI, preserving Telegram Markdown formatting.
    Also summarizes text if it exceeds Telegram limits:
    - Media caption limit: 1,024 characters (can be extended to 4,096 with sendMessage)
    - Regular message limit: 4,096 characters
    """
    logger.info(f"Enhance request received: parse_mode={payload.parse_mode}, message_length={len(payload.message) if payload.message else 0}")
    
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set")
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY environment variable is not set. Please set it in .env file or environment variables."
        )

    # OpenRouter API settings
    base_url = OPENROUTER_BASE_URL
    model = OPENROUTER_MODEL
    
    # Получаем заголовки для OpenRouter (HTTP-Referer и X-Title)
    openrouter_headers = get_openrouter_headers()
    
    logger.info(f"Using OpenRouter model: {model}, base_url: {base_url}")
    if openrouter_headers:
        logger.info(f"Using OpenRouter attribution headers: {list(openrouter_headers.keys())}")

    try:
        # Создаем OpenRouter клиент (используя OpenAI-совместимый API)
        # OpenRouter использует OpenAI-совместимый протокол
        try:
            client_kwargs = {
                "api_key": OPENROUTER_API_KEY,
                "base_url": base_url,
            }
            
            logger.info(f"Creating OpenRouter client with base_url: {base_url}")
            
            # Добавляем заголовки для OpenRouter (HTTP-Referer и X-Title)
            if openrouter_headers:
                client_kwargs["default_headers"] = openrouter_headers
                logger.info(f"Adding OpenRouter attribution headers: {openrouter_headers}")
            
            client = OpenAI(**client_kwargs)
        except TypeError as e:
            # Если возникает ошибка с параметрами, пробуем без заголовков
            logger.warning(f"Error creating OpenRouter client with headers: {e}, trying without headers")
            if "default_headers" in str(e):
                client = OpenAI(
                    api_key=OPENROUTER_API_KEY,
                    base_url=base_url,
                )
                logger.warning(f"OpenRouter headers will be ignored due to client initialization error")
            else:
                raise
        
        # Определяем инструкции в зависимости от parse_mode
        # На основе документации: https://core.telegram.org/api/entities
        if payload.parse_mode == "MarkdownV2":
            format_instructions = """
Используй Telegram Markdown форматирование (согласно https://core.telegram.org/api/entities):
- *жирный* для жирного текста (messageEntityBold) - выделять слово звездочками (одной с каждой стороны)
- _курсив_ для курсива (messageEntityItalic) - выделять слово нижним подчеркиванием
- __подчеркнутый__ для подчеркнутого (messageEntityUnderline)
- ~зачеркнутый~ для зачеркнутого (messageEntityStrike)
- `моноширинный` для моноширинного (messageEntityCode)
- не используй ** для жирного текста, используй *
- ```язык
код
``` для блока кода (messageEntityPre)
- [текст](URL) для ссылок
- Важно: экранируй специальные символы: _ * [ ] ( ) ~ ` > # + - = | { } . !
- Длина entities считается в UTF-16 code units (эмодзи и символы вне BMP считаются как 2 единицы)
- Пример: Привет друзья переделать в *Привет, друзья!*
"""
        elif payload.parse_mode == "HTML":
            format_instructions = """
Используй Telegram HTML форматирование (согласно https://core.telegram.org/api/entities):
- <b>жирный</b> или <strong>жирный</strong> для жирного текста (messageEntityBold)
- <i>курсив</i> или <em>курсив</em> для курсива (messageEntityItalic)
- <u>подчеркнутый</u> для подчеркнутого (messageEntityUnderline)
- <s>зачеркнутый</s>, <strike>зачеркнутый</strike> или <del>зачеркнутый</del> для зачеркнутого (messageEntityStrike)
- <code>моноширинный</code> для моноширинного (messageEntityCode)
- <pre language="язык">код</pre> для блока кода (messageEntityPre)
- <a href="URL">текст</a> для ссылок
- Поддерживаются вложенные entities
"""
        else:  # Markdown (default)
            format_instructions = """
Используй Telegram Markdown форматирование (согласно https://core.telegram.org/api/entities):
- *жирный* для жирного текста (messageEntityBold) - выделять слово звездочками (одной с каждой стороны)
- _курсив_ для курсива (messageEntityItalic)
- `моноширинный` для моноширинного (messageEntityCode) - выделять слово обратными апострофами (одного с каждой стороны)
- ```язык
код
``` для блока кода (messageEntityPre) - выделять слово обратными апострофами (одного с каждой стороны)
- [текст](URL) для ссылок - выделять слово квадратными скобками (одной с каждой стороны)
- Поддерживаются вложенные entities
- не используй ** для жирного текста, используй *
"""

        # Определяем лимиты Telegram
        # Media caption: 1,024 символа (можно до 4,096 с sendMessage)
        # Regular message: 4,096 символов
        message_length = len(payload.message)
        MEDIA_CAPTION_LIMIT = 1024
        MESSAGE_LIMIT = 4096
        
        # Определяем, нужно ли сокращать текст
        needs_summarization = message_length > MESSAGE_LIMIT
        needs_caption_summarization = message_length > MEDIA_CAPTION_LIMIT
        
        if needs_summarization or needs_caption_summarization:
            # Определяем целевой лимит
            if needs_summarization:
                target_limit = MESSAGE_LIMIT
                limit_type = "обычного сообщения (4,096 символов)"
            else:
                target_limit = MEDIA_CAPTION_LIMIT
                limit_type = "подписи к медиа (1,024 символа, можно расширить до 4,096 с sendMessage)"
            
            logger.info(f"Text exceeds Telegram limit ({message_length} > {target_limit}). Will summarize to fit {limit_type}")
            
            system_prompt = f"""Ты эксперт по написанию и суммаризации сообщений для Telegram.
Твоя задача - сократить и улучшить текст сообщения, сохранив все ключевые моменты и форматирование.

{format_instructions}

Важно:
- Сократи текст до {target_limit} символов или меньше
- Сохрани все важные ссылки и их форматирование
- Сохрани ключевую информацию и смысл сообщения
- Используй форматирование для выделения важных моментов (жирный, курсив, подчеркивание)
- Делай текст более компактным, но читаемым
- Если в тексте уже есть форматирование, сохрани его
- Поддерживай вложенные entities (например, жирный текст внутри ссылки)
- Убедись, что все теги правильно закрыты и валидны для Telegram API
- Не используй ** для жирного текста, используй *
- Результат должен быть не более {target_limit} символов"""
            
            user_prompt = f"Сократи и улучши следующий текст сообщения, сохранив все ключевые моменты и форматирование. Текст должен быть не более {target_limit} символов:\n\n{payload.message}"
        else:
            system_prompt = f"""Ты эксперт по написанию привлекательных сообщений для Telegram.
Твоя задача - улучшить текст сообщения, сделав его более интересным, структурированным и привлекательным для читателя.

{format_instructions}

Важно:
- Сохраняй все ссылки и их форматирование
- Используй форматирование для выделения важных моментов (жирный, курсив, подчеркивание)
- Делай текст более читаемым с помощью списков, абзацев и форматирования
- Сохраняй общий смысл и тон сообщения
- Не добавляй лишнюю информацию, только улучшай существующий текст
- Если в тексте уже есть форматирование, сохрани его и улучши
- Поддерживай вложенные entities (например, жирный текст внутри ссылки)
- Убедись, что все теги правильно закрыты и валидны для Telegram API
- Не используй ** для жирного текста, используй *, да и в целом не используй ** для форматирования, используй *"""
            
            user_prompt = f"Улучши следующий текст сообщения:\n\n{payload.message}"

        def create_completion():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
            )
        
        response = await run_in_threadpool(create_completion)
        
        logger.info(f"OpenRouter API response received, choices count: {len(response.choices) if response.choices else 0}")

        if not response.choices or len(response.choices) == 0:
            raise HTTPException(
                status_code=500,
                detail="OpenRouter API returned no choices in response"
            )
        
        enhanced_text = response.choices[0].message.content.strip()
        
        if not enhanced_text:
            raise HTTPException(
                status_code=500,
                detail="OpenRouter API returned empty enhanced text"
            )
        
        # Проверяем, что результат не превышает лимиты
        final_length = len(enhanced_text)
        if needs_summarization and final_length > MESSAGE_LIMIT:
            logger.warning(f"Enhanced text still exceeds message limit ({final_length} > {MESSAGE_LIMIT}), truncating...")
            # Обрезаем до лимита, стараясь не обрезать посередине форматирования
            enhanced_text = enhanced_text[:MESSAGE_LIMIT]
            # Пытаемся найти последний закрывающий тег или символ форматирования
            if payload.parse_mode == "HTML":
                # Ищем последний закрывающий тег
                last_tag_end = enhanced_text.rfind(">")
                if last_tag_end > MESSAGE_LIMIT - 100:  # Если тег близко к концу
                    enhanced_text = enhanced_text[:last_tag_end + 1]
            elif payload.parse_mode in ["Markdown", "MarkdownV2"]:
                # Ищем последний закрывающий символ форматирования
                for char in ["*", "_", "`", "]"]:
                    last_pos = enhanced_text.rfind(char)
                    if last_pos > MESSAGE_LIMIT - 50:
                        enhanced_text = enhanced_text[:last_pos + 1]
                        break
        elif needs_caption_summarization and final_length > MEDIA_CAPTION_LIMIT:
            logger.warning(f"Enhanced text still exceeds media caption limit ({final_length} > {MEDIA_CAPTION_LIMIT}), truncating...")
            # Аналогичная логика для caption
            enhanced_text = enhanced_text[:MEDIA_CAPTION_LIMIT]
            if payload.parse_mode == "HTML":
                last_tag_end = enhanced_text.rfind(">")
                if last_tag_end > MEDIA_CAPTION_LIMIT - 100:
                    enhanced_text = enhanced_text[:last_tag_end + 1]
            elif payload.parse_mode in ["Markdown", "MarkdownV2"]:
                for char in ["*", "_", "`", "]"]:
                    last_pos = enhanced_text.rfind(char)
                    if last_pos > MEDIA_CAPTION_LIMIT - 50:
                        enhanced_text = enhanced_text[:last_pos + 1]
                        break
        
        logger.info(f"Enhanced message length: {len(enhanced_text)} (original: {message_length})")
        
        return {
            "enhanced_message": enhanced_text,
            "original_length": message_length,
            "enhanced_length": len(enhanced_text),
            "was_summarized": needs_summarization or needs_caption_summarization
        }
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to enhance message: {exc}", exc_info=True)
        error_msg = str(exc)
        # Улучшаем сообщение об ошибке для пользователя
        if "api key" in error_msg.lower() or "authentication" in error_msg.lower():
            error_msg = "Invalid API key. Please check your OPENROUTER_API_KEY."
        elif "rate limit" in error_msg.lower():
            error_msg = "Rate limit exceeded. Please try again later."
        elif "insufficient_quota" in error_msg.lower():
            error_msg = "Insufficient quota. Please check your OpenRouter account balance."
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enhance message: {error_msg}"
        ) from exc


@app.post("/api/send", tags=["messaging"])
async def send_broadcast(
    token: Optional[str] = Form(None),
    selected_bots: Optional[str] = Form(None),
    message: str = Form(...),
    parse_mode: str = Form("Markdown"),
    disable_web_page_preview: bool = Form(False),
    inline_keyboard: Optional[str] = Form(None),
    attach_message_to_first_photo: bool = Form(False),
    extra_api_params: Optional[str] = Form(None),
    csv_file: Optional[UploadFile] = File(None),
    csv_file_name: Optional[str] = Form(None),
    photos: List[UploadFile] = File(default=[]),
    videos: List[UploadFile] = File(default=[]),
):
    # Determine which tokens to use
    tokens_to_use: List[str] = []
    
    if selected_bots:
        try:
            selected_tokens = json.loads(selected_bots)
            if isinstance(selected_tokens, list):
                tokens_to_use = [t for t in selected_tokens if t and t.strip()]
            elif isinstance(selected_tokens, str):
                tokens_to_use = [selected_tokens] if selected_tokens.strip() else []
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="selected_bots must be valid JSON array")
    
    # Fall back to single token if no selected_bots provided
    if not tokens_to_use:
        if token and token.strip():
            tokens_to_use = [token.strip()]
        else:
            raise HTTPException(status_code=400, detail="Bot token is required (either 'token' or 'selected_bots')")
    
    if not tokens_to_use:
        raise HTTPException(status_code=400, detail="At least one bot token is required")

    # Handle CSV file: either uploaded or selected from data directory
    csv_bytes: bytes
    if csv_file_name:
        # Load from data directory
        possible_paths = [
            Path(__file__).resolve().parents[2] / "data" / csv_file_name,
            Path("/app/data") / csv_file_name,
            Path("data") / csv_file_name,
        ]
        
        csv_path = None
        for path in possible_paths:
            if path.exists() and path.is_file():
                csv_path = path
                break
        
        if not csv_path:
            raise HTTPException(
                status_code=400,
                detail=f"CSV file '{csv_file_name}' not found in data directory"
            )
        
        try:
            async with aiofiles.open(csv_path, "rb") as f:
                csv_bytes = await f.read()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read CSV file from data directory: {exc}"
            ) from exc
    elif csv_file:
        # Use uploaded file
        csv_bytes = await csv_file.read()
    else:
        raise HTTPException(
            status_code=400,
            detail="Either upload a CSV file or select one from the data directory"
        )
    
    if not csv_bytes:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    # Note: We'll filter blocked users per bot token later, after we know which bots are being used
    # For now, load all chat IDs without filtering (filtering will happen per bot)
    try:
        chat_ids = await run_in_threadpool(load_chat_ids_from_csv, csv_bytes, filter_blocked=False)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}") from exc

    if not chat_ids:
        raise HTTPException(
            status_code=400,
            detail="No valid telegram_id values found in the CSV file",
        )

    photo_objects: List[TelegramPhoto] = []
    if photos:
        logger.info(f"Received {len(photos)} photo(s)")
        for upload in photos:
            content = await upload.read()
            if not content:
                logger.warning(f"Empty photo file: {upload.filename}")
                continue
            logger.info(f"Processing photo: {upload.filename}, size: {len(content)} bytes")
            photo_objects.append(
                TelegramPhoto(
                    filename=upload.filename or "photo.jpg",
                    content=content,
                    content_type=upload.content_type or "image/jpeg",
                )
            )
    else:
        logger.info("No photos received")

    video_objects: List[TelegramVideo] = []
    if videos:
        logger.info(f"Received {len(videos)} video(s)")
        for upload in videos:
            content = await upload.read()
            if not content:
                logger.warning(f"Empty video file: {upload.filename}")
                continue
            logger.info(f"Processing video: {upload.filename}, size: {len(content)} bytes")
            video_objects.append(
                TelegramVideo(
                    filename=upload.filename or "video.mp4",
                    content=content,
                    content_type=upload.content_type or "video/mp4",
                )
            )
    else:
        logger.info("No videos received")

    keyboard_rows: Optional[List[List[InlineButton]]] = None
    if inline_keyboard:
        try:
            loaded_keyboard = json.loads(inline_keyboard)
            keyboard_rows = [
                [InlineButton(**button) for button in row] for row in loaded_keyboard
            ]
        except (json.JSONDecodeError, TypeError, ValidationError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid inline keyboard definition: {exc}",
            ) from exc

    extra_params = {}
    if extra_api_params:
        try:
            extra_params = json.loads(extra_api_params)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"extra_api_params must be valid JSON: {exc}",
            ) from exc

    normalized_parse_mode = None if parse_mode == "None" else parse_mode

    # Send to all selected bots
    all_results = []
    total_delivered = 0
    total_failed = []
    total_count = 0
    
    for bot_token in tokens_to_use:
        # Filter out blocked users for this specific bot token
        blocked_users_for_bot = await load_blocked_users_async(bot_token) if bot_token else set()
        filtered_chat_ids = [cid for cid in chat_ids if cid not in blocked_users_for_bot]
        
        if blocked_users_for_bot:
            logger.info(f"Filtered out {len(blocked_users_for_bot)} blocked user(s) for bot token {bot_token[:20]}...")
        
        config = TelegramBroadcastConfig(
            token=bot_token,
            message=message,
            parse_mode=normalized_parse_mode,
            disable_web_page_preview=disable_web_page_preview,
            inline_keyboard=keyboard_rows,
            photos=photo_objects if photo_objects else None,
            videos=video_objects if video_objects else None,
            attach_message_to_first_photo=attach_message_to_first_photo,
            attach_message_to_first_video=attach_message_to_first_photo,  # Используем тот же флаг для видео
            extra_api_params=extra_params,
        )

        sender = TelegramSender(token=bot_token)
        # Используем асинхронную версию для параллельной отправки
        summary = await sender.broadcast_async(filtered_chat_ids, config)
        
        total_count += summary.total
        total_delivered += summary.delivered
        total_failed.extend(summary.failed)
        
        # Extract blocked users for this specific bot
        # These are users who have blocked this bot, deactivated their account, or chat not found
        blocked_user_ids_for_bot = set()
        for report in summary.failed:
            if report.error and is_blocked_user_error(report.error):
                blocked_user_ids_for_bot.add(str(report.chat_id))
        
        # Save blocked users to file for this specific bot token
        if blocked_user_ids_for_bot:
            try:
                await save_blocked_users_async(bot_token, blocked_user_ids_for_bot)
                logger.info(f"Saved {len(blocked_user_ids_for_bot)} blocked user(s) for bot token {bot_token[:20]}...")
            except Exception as e:
                logger.error(f"Failed to save blocked users for bot token: {e}", exc_info=True)
        
        all_results.append({
            "token": bot_token[:20] + "..." if len(bot_token) > 20 else bot_token,
            "total": summary.total,
            "delivered": summary.delivered,
            "failed_count": len(summary.failed),
            "success": summary.success,
            "blocked_users_saved": len(blocked_user_ids_for_bot),
        })

    response = {
        "total": total_count,
        "delivered": total_delivered,
        "failed": [
            {"chat_id": report.chat_id, "error": report.error}
            for report in total_failed
        ],
        "success": len(total_failed) == 0,
        "bots": all_results,
    }

    status_code = 200 if response["success"] else 207
    return JSONResponse(content=response, status_code=status_code)


@app.get("/")
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Frontend not found. Build assets under the frontend directory."}

