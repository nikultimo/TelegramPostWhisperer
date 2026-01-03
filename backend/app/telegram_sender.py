from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable, List, Optional, Sequence, Tuple, Union

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InlineButton:
    text: str
    url: Optional[str] = None
    callback_data: Optional[str] = None

    def to_dict(self) -> dict:
        data = {"text": self.text}
        if self.url:
            data["url"] = self.url
        if self.callback_data:
            data["callback_data"] = self.callback_data
        return data


@dataclass(slots=True)
class TelegramPhoto:
    filename: str
    content: bytes
    content_type: str = "image/jpeg"


@dataclass(slots=True)
class TelegramVideo:
    filename: str
    content: bytes
    content_type: str = "video/mp4"
    duration: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    thumbnail: Optional[TelegramPhoto] = None


@dataclass(slots=True)
class DeliveryReport:
    chat_id: Union[int, str]
    ok: bool
    error: Optional[str] = None


@dataclass(slots=True)
class BroadcastSummary:
    total: int
    delivered: int
    failed: List[DeliveryReport] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.delivered == self.total


@dataclass(slots=True)
class TelegramBroadcastConfig:
    token: str
    message: str
    parse_mode: str = "Markdown"
    disable_web_page_preview: bool = False
    inline_keyboard: Optional[List[List[InlineButton]]] = None
    photos: Optional[Sequence[TelegramPhoto]] = None
    videos: Optional[Sequence[TelegramVideo]] = None
    attach_message_to_first_photo: bool = False
    attach_message_to_first_video: bool = False
    extra_api_params: dict = field(default_factory=dict)


def load_chat_ids_from_csv(
    csv_source: Union[str, Path, bytes, IO[bytes], IO[str]],
    *,
    column_name: str = "telegram_id",
    encoding: str = "utf-8",
    filter_blocked: bool = True,
    bot_token: Optional[str] = None,
) -> List[str]:
    """
    Load chat identifiers from a CSV file-like input.

    The CSV may contain a column named ``telegram_id`` or provide the IDs in the first column.
    Empty rows and malformed entries are skipped.
    
    Args:
        csv_source: CSV file path, bytes, or file-like object
        column_name: Name of the column containing telegram IDs
        encoding: Text encoding for the CSV file
        filter_blocked: If True, filter out blocked users from blocked_users.csv
        bot_token: Bot token to filter blocked users for (required if filter_blocked=True)
    
    Returns:
        List of chat IDs (as strings), with blocked users filtered out if filter_blocked=True
    """

    if isinstance(csv_source, (str, Path)):
        with open(csv_source, "rb") as fh:
            content = fh.read()
    elif isinstance(csv_source, bytes):
        content = csv_source
    elif hasattr(csv_source, "read"):
        raw = csv_source.read()
        content = raw.encode(encoding) if isinstance(raw, str) else raw
    else:
        raise TypeError("Unsupported csv_source type")

    text_stream = io.StringIO(content.decode(encoding))

    reader = csv.reader(text_stream)
    rows = list(reader)
    if not rows:
        return []

    header = [cell.strip() for cell in rows[0]]
    data_rows = rows[1:] if _looks_like_header(header, column_name) else rows

    # Load blocked users if filtering is enabled
    blocked_users: set[str] = set()
    if filter_blocked:
        if not bot_token:
            logger.warning("filter_blocked=True but bot_token not provided, skipping blocked user filtering")
        else:
            blocked_users = load_blocked_users(bot_token)
            if blocked_users:
                logger.info(f"Filtering out {len(blocked_users)} blocked user(s) for bot token")

    ids_set: set[str] = set()  # Use set to automatically remove duplicates
    column_idx = (
        header.index(column_name) if header and column_name in header else 0
    )

    filtered_count = 0
    duplicate_count = 0
    for row in data_rows:
        if not row:
            continue

        try:
            raw_value = row[column_idx].strip()
        except IndexError:
            logger.debug("Skipping row without expected column: %s", row)
            continue

        if not raw_value:
            continue

        # Keep original string to preserve large integers and negative IDs
        normalized = raw_value.replace(" ", "")
        
        # Filter out blocked users
        if filter_blocked and normalized in blocked_users:
            filtered_count += 1
            continue
        
        # Check for duplicates
        if normalized in ids_set:
            duplicate_count += 1
            continue
        
        ids_set.add(normalized)

    if filter_blocked and filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} blocked user(s) from CSV")
    
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate user(s) from CSV")

    # Convert set to list to preserve order (Python 3.7+ maintains insertion order for sets)
    return list(ids_set)


def _looks_like_header(header: Sequence[str], column_name: str) -> bool:
    if not header:
        return False
    normalized = [cell.lower().strip() for cell in header if cell]
    return column_name.lower() in normalized


def _get_token_hash(token: str) -> str:
    """
    Generate a hash of the bot token for use in filenames.
    Uses SHA256 and returns first 16 characters for readability.
    """
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def get_blocked_users_file_path(token: str) -> Path:
    """
    Get the path to the blocked users CSV file for a specific bot token.
    Tries multiple possible locations (project root/data/blocked_users, Docker /app/data/blocked_users, current directory/data/blocked_users).
    
    Args:
        token: Bot token to generate unique filename
    
    Returns:
        Path to the blocked users CSV file for this bot
    """
    token_hash = _get_token_hash(token)
    filename = f"blocked_users_{token_hash}.csv"
    
    possible_paths = [
        Path(__file__).resolve().parents[2] / "data" / "blocked_users" / filename,  # Project root/data/blocked_users
        Path("/app/data") / "blocked_users" / filename,  # Docker container path
        Path("data") / "blocked_users" / filename,  # Current working directory/data/blocked_users
    ]
    
    for path in possible_paths:
        if path.parent.exists():
            return path
    
    # Return the first path if none exist (will create directory if needed)
    return possible_paths[0]


def load_blocked_users(token: str) -> set[str]:
    """
    Load blocked user IDs from the blocked_users CSV file for a specific bot token.
    
    Args:
        token: Bot token to load blocked users for
    
    Returns:
        Set of blocked user IDs (as strings).
    """
    blocked_file = get_blocked_users_file_path(token)
    
    if not blocked_file.exists():
        logger.debug(f"Blocked users file not found at {blocked_file}, returning empty set")
        return set()
    
    try:
        blocked_ids = set()
        with open(blocked_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            if not rows:
                return set()
            
            # Check if first row is a header
            header = [cell.strip() for cell in rows[0]]
            data_rows = rows[1:] if _looks_like_header(header, "telegram_id") else rows
            
            column_idx = (
                header.index("telegram_id") if header and "telegram_id" in header else 0
            )
            
            for row in data_rows:
                if not row:
                    continue
                try:
                    raw_value = row[column_idx].strip()
                    if raw_value:
                        normalized = raw_value.replace(" ", "")
                        blocked_ids.add(normalized)
                except (IndexError, ValueError):
                    continue
        
        logger.info(f"Loaded {len(blocked_ids)} blocked user(s) from {blocked_file}")
        return blocked_ids
    except Exception as e:
        logger.error(f"Failed to load blocked users from {blocked_file}: {e}")
        return set()


def is_blocked_user_error(error_text: Optional[str]) -> bool:
    """
    Determine if an error message indicates that a user is blocked or deactivated.
    
    Args:
        error_text: Error message from Telegram API
    
    Returns:
        True if the error indicates a blocked/deactivated user, False otherwise
    """
    if not error_text:
        return False
    
    error_lower = error_text.lower()
    
    # Connection errors should not be considered as blocked users (these are temporary)
    connection_patterns = [
        "connection error",
        "timeout",
        "cannot connect",
        "ssl:default",
        "network",
        "retrying",
        "attempt",
    ]
    
    # Check if it's a connection error first (these are temporary, not permanent blocks)
    for pattern in connection_patterns:
        if pattern in error_lower:
            return False
    
    # Check for various blocked/deactivated user error patterns
    # These indicate permanent issues that should be saved to avoid retrying
    blocked_patterns = [
        "user_is_blocked",
        "user is blocked",
        "user is deactivated",
        "chat not found",
        "forbidden: user is deactivated",
        "forbidden: user_is_blocked",
    ]
    
    # Check if it matches any blocked user pattern
    for pattern in blocked_patterns:
        if pattern in error_lower:
            return True
    
    return False


def save_blocked_users(token: str, blocked_ids: set[str]) -> None:
    """
    Save blocked user IDs to the blocked_users CSV file for a specific bot token.
    Appends new blocked users to the existing file (if any).
    
    Args:
        token: Bot token to save blocked users for
        blocked_ids: Set of blocked user IDs to save
    """
    if not blocked_ids:
        logger.debug("No blocked users to save")
        return
    
    blocked_file = get_blocked_users_file_path(token)
    
    # Load existing blocked users to avoid duplicates
    existing_blocked = load_blocked_users(token)
    
    # Merge with new blocked users
    all_blocked = existing_blocked | blocked_ids
    
    # Only save if there are new blocked users
    new_blocked = blocked_ids - existing_blocked
    if not new_blocked:
        logger.debug("No new blocked users to save")
        return
    
    try:
        # Ensure directory exists
        blocked_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write all blocked users (existing + new)
        with open(blocked_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["telegram_id"])  # Header
            for user_id in sorted(all_blocked, key=lambda x: (len(x), x)):
                writer.writerow([user_id])
        
        logger.info(f"Saved {len(new_blocked)} new blocked user(s) to {blocked_file} (total: {len(all_blocked)})")
    except OSError as e:
        # Handle read-only filesystem gracefully
        if e.errno == 30:  # Read-only file system
            logger.warning(
                f"Cannot save blocked users to {blocked_file}: filesystem is read-only. "
                f"{len(new_blocked)} blocked user(s) detected but not persisted. "
                f"Consider mounting /app/data as writable or using a writable volume."
            )
        else:
            logger.error(f"Failed to save blocked users to {blocked_file}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to save blocked users to {blocked_file}: {e}", exc_info=True)


class TelegramSender:
    api_host = "https://api.telegram.org"

    def __init__(self, token: str, *, session: Optional[requests.Session] = None):
        self.token = token
        if session is None:
            session = requests.Session()
            # Configure retry strategy for SSL and connection errors
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST", "GET"],
                raise_on_status=False,
            )
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,
                pool_maxsize=20,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
        self.session = session

    @property
    def base_url(self) -> str:
        return f"{self.api_host}/bot{self.token}"

    def broadcast(
        self,
        chat_ids: Iterable[Union[int, str]],
        config: TelegramBroadcastConfig,
    ) -> BroadcastSummary:
        # Дедупликация chat_ids для предотвращения повторной отправки одному пользователю
        seen = set()
        unique_chat_ids = []
        duplicates_count = 0
        for chat_id in chat_ids:
            chat_id_str = str(chat_id)
            if chat_id_str not in seen:
                seen.add(chat_id_str)
                unique_chat_ids.append(chat_id)
            else:
                duplicates_count += 1
        
        if duplicates_count > 0:
            logger.warning(f"Found {duplicates_count} duplicate chat_id(s), removed to prevent duplicate messages")
        
        chat_ids = unique_chat_ids
        failed: List[DeliveryReport] = []
        delivered = 0

        photos = list(config.photos or [])
        videos = list(config.videos or [])

        for chat_id in chat_ids:
            if videos:
                # Если есть видео, отправляем видео (с подписью или без)
                # Inline-кнопки добавляем к последнему видео, если текст не прикреплен к первому
                attach_to_last = not config.attach_message_to_first_video and config.inline_keyboard
                media_failures = self._send_videos(
                    chat_id,
                    videos,
                    config.attach_message_to_first_video,
                    config.message if config.attach_message_to_first_video else "",
                    config.parse_mode if config.attach_message_to_first_video else None,
                    config.inline_keyboard if (config.attach_message_to_first_video or attach_to_last) else None,
                    attach_to_last,
                )
                if media_failures:
                    failed.extend(media_failures)
                else:
                    delivered += 1
                
                # Если текст не был прикреплен к первому видео, отправляем его отдельно
                # Но без inline-кнопок, если они уже были на последнем видео
                if not config.attach_message_to_first_video and config.message:
                    if attach_to_last:
                        # Создаем новый конфиг без inline-кнопок для текстового сообщения
                        message_config = TelegramBroadcastConfig(
                            token=config.token,
                            message=config.message,
                            parse_mode=config.parse_mode,
                            disable_web_page_preview=config.disable_web_page_preview,
                            inline_keyboard=None,  # Кнопки уже на последнем видео
                            photos=None,
                            videos=None,
                            attach_message_to_first_photo=False,
                            attach_message_to_first_video=False,
                            extra_api_params=config.extra_api_params,
                        )
                    else:
                        message_config = config
                    message_result = self._send_message(chat_id, message_config)
                    if not message_result.ok:
                        failed.append(message_result)
            elif photos:
                # Если есть фото, отправляем фото (с подписью или без)
                # Inline-кнопки добавляем к последнему фото, если текст не прикреплен к первой
                attach_to_last = not config.attach_message_to_first_photo and config.inline_keyboard
                media_failures = self._send_photos(
                    chat_id,
                    photos,
                    config.attach_message_to_first_photo,
                    config.message if config.attach_message_to_first_photo else "",
                    config.parse_mode if config.attach_message_to_first_photo else None,
                    config.inline_keyboard if (config.attach_message_to_first_photo or attach_to_last) else None,
                    attach_to_last,
                )
                if media_failures:
                    failed.extend(media_failures)
                else:
                    delivered += 1
                
                # Если текст не был прикреплен к первой фото, отправляем его отдельно
                # Но без inline-кнопок, если они уже были на последнем фото
                if not config.attach_message_to_first_photo and config.message:
                    if attach_to_last:
                        # Создаем новый конфиг без inline-кнопок для текстового сообщения
                        message_config = TelegramBroadcastConfig(
                            token=config.token,
                            message=config.message,
                            parse_mode=config.parse_mode,
                            disable_web_page_preview=config.disable_web_page_preview,
                            inline_keyboard=None,  # Кнопки уже на последнем фото
                            photos=None,
                            attach_message_to_first_photo=False,
                            extra_api_params=config.extra_api_params,
                        )
                    else:
                        message_config = config
                    message_result = self._send_message(chat_id, message_config)
                    if not message_result.ok:
                        failed.append(message_result)
            else:
                # Если фото нет, отправляем только текстовое сообщение
                message_result = self._send_message(chat_id, config)
                if not message_result.ok:
                    failed.append(message_result)
                else:
                    delivered += 1

        return BroadcastSummary(total=len(chat_ids), delivered=delivered, failed=failed)

    async def broadcast_async(
        self,
        chat_ids: Iterable[Union[int, str]],
        config: TelegramBroadcastConfig,
        *,
        max_concurrent: int = 50,
        rate_limit_per_second: float = 30.0,
    ) -> BroadcastSummary:
        """
        Асинхронная версия broadcast с параллельной отправкой сообщений.
        
        Args:
            chat_ids: Список ID чатов для рассылки
            config: Конфигурация рассылки
            max_concurrent: Максимальное количество одновременных запросов (по умолчанию 50)
            rate_limit_per_second: Максимальное количество запросов в секунду (по умолчанию 30)
        """
        # Дедупликация chat_ids для предотвращения повторной отправки одному пользователю
        seen = set()
        unique_chat_ids = []
        duplicates_count = 0
        for chat_id in chat_ids:
            chat_id_str = str(chat_id)
            if chat_id_str not in seen:
                seen.add(chat_id_str)
                unique_chat_ids.append(chat_id)
            else:
                duplicates_count += 1
        
        if duplicates_count > 0:
            logger.warning(f"Found {duplicates_count} duplicate chat_id(s), removed to prevent duplicate messages")
        
        chat_ids = unique_chat_ids
        if not chat_ids:
            return BroadcastSummary(total=0, delivered=0, failed=[])

        photos = list(config.photos or [])
        videos = list(config.videos or [])
        
        # Для малых батчей (<= 50 пользователей) не применяем строгий rate limiting
        # Telegram API позволяет небольшие всплески запросов
        use_rate_limiting = len(chat_ids) > 50
        
        # Семафор для ограничения количества одновременных запросов
        semaphore = asyncio.Semaphore(max_concurrent)
        
        refiller_task = None
        token_queue = None
        
        if use_rate_limiting:
            # Улучшенный rate limiter с использованием очереди токенов
            # Это позволяет избежать блокировки всех запросов одним lock
            min_interval = 1.0 / rate_limit_per_second
            # Увеличиваем размер очереди для лучшей пропускной способности
            queue_size = int(rate_limit_per_second * 3)
            token_queue = asyncio.Queue(maxsize=queue_size)
            
            # Заполняем очередь токенами
            for _ in range(queue_size):
                token_queue.put_nowait(None)
            
            # Фоновая задача для пополнения токенов
            async def token_refiller():
                """Пополняет очередь токенов с нужной частотой"""
                while True:
                    await asyncio.sleep(min_interval)
                    try:
                        token_queue.put_nowait(None)
                    except asyncio.QueueFull:
                        pass  # Очередь уже полная, пропускаем
            
            # Запускаем refiller как фоновую задачу
            refiller_task = asyncio.create_task(token_refiller())
        
        async def rate_limited_request():
            """Ожидание для соблюдения rate limit через токены"""
            if use_rate_limiting and token_queue:
                await token_queue.get()  # Получаем токен из очереди
            # Для малых батчей просто пропускаем rate limiting
        
        async def send_to_chat(chat_id: Union[int, str]) -> Optional[DeliveryReport]:
            """Отправка сообщения одному чату"""
            async with semaphore:
                await rate_limited_request()
                
                try:
                    if videos:
                        attach_to_last = not config.attach_message_to_first_video and config.inline_keyboard
                        media_failures = await self._send_videos_async(
                            chat_id,
                            videos,
                            config.attach_message_to_first_video,
                            config.message if config.attach_message_to_first_video else "",
                            config.parse_mode if config.attach_message_to_first_video else None,
                            config.inline_keyboard if (config.attach_message_to_first_video or attach_to_last) else None,
                            attach_to_last,
                        )
                        if media_failures:
                            return media_failures[0] if media_failures else None
                        
                        if not config.attach_message_to_first_video and config.message:
                            if attach_to_last:
                                message_config = TelegramBroadcastConfig(
                                    token=config.token,
                                    message=config.message,
                                    parse_mode=config.parse_mode,
                                    disable_web_page_preview=config.disable_web_page_preview,
                                    inline_keyboard=None,
                                    photos=None,
                                    videos=None,
                                    attach_message_to_first_photo=False,
                                    attach_message_to_first_video=False,
                                    extra_api_params=config.extra_api_params,
                                )
                            else:
                                message_config = config
                            message_result = await self._send_message_async(chat_id, message_config)
                            if not message_result.ok:
                                return message_result
                        return None
                    elif photos:
                        attach_to_last = not config.attach_message_to_first_photo and config.inline_keyboard
                        media_failures = await self._send_photos_async(
                            chat_id,
                            photos,
                            config.attach_message_to_first_photo,
                            config.message if config.attach_message_to_first_photo else "",
                            config.parse_mode if config.attach_message_to_first_photo else None,
                            config.inline_keyboard if (config.attach_message_to_first_photo or attach_to_last) else None,
                            attach_to_last,
                        )
                        if media_failures:
                            return media_failures[0] if media_failures else None
                        
                        if not config.attach_message_to_first_photo and config.message:
                            if attach_to_last:
                                message_config = TelegramBroadcastConfig(
                                    token=config.token,
                                    message=config.message,
                                    parse_mode=config.parse_mode,
                                    disable_web_page_preview=config.disable_web_page_preview,
                                    inline_keyboard=None,
                                    photos=None,
                                    attach_message_to_first_photo=False,
                                    extra_api_params=config.extra_api_params,
                                )
                            else:
                                message_config = config
                            message_result = await self._send_message_async(chat_id, message_config)
                            if not message_result.ok:
                                return message_result
                        return None
                    else:
                        message_result = await self._send_message_async(chat_id, config)
                        if not message_result.ok:
                            return message_result
                        return None
                except Exception as e:
                    logger.error(f"Unexpected error sending to {chat_id}: {e}", exc_info=True)
                    return DeliveryReport(chat_id=chat_id, ok=False, error=f"Unexpected error: {e}")
        
        # Создаем сессию aiohttp с оптимизированными настройками
        connector = aiohttp.TCPConnector(
            limit=max_concurrent * 3,  # Увеличиваем лимит соединений для лучшей параллельности
            limit_per_host=max_concurrent * 2,  # Увеличиваем лимит на хост
            ttl_dns_cache=600,  # Увеличиваем кеш DNS
            force_close=False,  # Переиспользуем соединения
            enable_cleanup_closed=True,  # Автоматическая очистка закрытых соединений
        )
        timeout = aiohttp.ClientTimeout(total=300, connect=30, sock_read=60)  # Увеличиваем таймауты для больших файлов
        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                self._async_session = session
                
                # Запускаем все задачи параллельно
                tasks = [send_to_chat(chat_id) for chat_id in chat_ids]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Обрабатываем результаты
                failed: List[DeliveryReport] = []
                delivered = 0
                
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Task raised exception: {result}", exc_info=True)
                        continue
                    if result is not None:
                        failed.append(result)
                    else:
                        delivered += 1
                
                return BroadcastSummary(total=len(chat_ids), delivered=delivered, failed=failed)
        finally:
            # Останавливаем фоновую задачу пополнения токенов (если она была запущена)
            if refiller_task:
                refiller_task.cancel()
                try:
                    await refiller_task
                except asyncio.CancelledError:
                    pass

    def _send_message(
        self,
        chat_id: Union[int, str],
        config: TelegramBroadcastConfig,
    ) -> DeliveryReport:
        payload = {
            "chat_id": chat_id,
            "text": config.message,
            "disable_web_page_preview": config.disable_web_page_preview,
        }

        if config.parse_mode:
            payload["parse_mode"] = config.parse_mode

        if config.inline_keyboard:
            payload["reply_markup"] = {
                "inline_keyboard": [
                    [button.to_dict() for button in row] for row in config.inline_keyboard
                ]
            }

        if config.extra_api_params:
            payload.update(config.extra_api_params)

        response = self.session.post(
            f"{self.base_url}/sendMessage",
            json=payload,
            timeout=15,
        )

        if response.ok:
            return DeliveryReport(chat_id=chat_id, ok=True)

        try:
            details = response.json()
            error_text = details.get("description", response.text)
        except json.JSONDecodeError:
            error_text = response.text

        logger.error("Failed to send message to %s: %s", chat_id, error_text)
        return DeliveryReport(chat_id=chat_id, ok=False, error=error_text)

    def _send_photos(
        self,
        chat_id: Union[int, str],
        photos: Sequence[TelegramPhoto],
        attach_message: bool,
        message: str,
        parse_mode: Optional[str],
        inline_keyboard: Optional[List[List[InlineButton]]],
        attach_keyboard_to_last: bool = False,
    ) -> List[DeliveryReport]:
        failures: List[DeliveryReport] = []

        if not photos:
            return failures

        # Если одно фото, используем sendPhoto (для обратной совместимости)
        if len(photos) == 1:
            photo = photos[0]
            data = {"chat_id": chat_id}
            
            if attach_message:
                data["caption"] = message
                if parse_mode:
                    data["parse_mode"] = parse_mode
            
            if inline_keyboard:
                try:
                    keyboard_dict = {
                        "inline_keyboard": [
                            [button.to_dict() for button in row] for row in inline_keyboard
                        ]
                    }
                    keyboard_json = json.dumps(keyboard_dict, separators=(',', ':'), ensure_ascii=False)
                    json.loads(keyboard_json)  # Валидация
                    logger.debug(f"Sending reply_markup JSON: {keyboard_json[:200]}...")
                    data["reply_markup"] = keyboard_json
                except (TypeError, ValueError) as e:
                    logger.error(f"Failed to serialize inline keyboard: {e}")

            files = {
                "photo": (
                    photo.filename or "photo.jpg",
                    photo.content,
                    photo.content_type or "image/jpeg",
                )
            }

            # Увеличиваем таймаут для больших фото
            photo_size_mb = len(photo.content) / (1024 * 1024)
            timeout = 120 if photo_size_mb > 5 else 60 if photo_size_mb > 1 else 30
            
            # Retry logic with exponential backoff for SSL/connection errors
            max_retries = 3
            retry_delay = 1
            last_exception = None
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = self.session.post(
                        f"{self.base_url}/sendPhoto",
                        data=data,
                        files=files,
                        timeout=timeout,
                    )
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.SSLError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(
                            "Photo send attempt %d/%d failed for chat_id %s: %s. Retrying in %d seconds...",
                            attempt + 1, max_retries, chat_id, e, wait_time
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error("Timeout or connection error sending photo to %s after %d attempts: %s", chat_id, max_retries, e)
                        failures.append(
                            DeliveryReport(chat_id=chat_id, ok=False, error=f"Timeout or connection error after {max_retries} attempts: {e}")
                        )
                        return failures
            
            if last_exception and response is None:
                failures.append(
                    DeliveryReport(chat_id=chat_id, ok=False, error=f"Timeout or connection error: {last_exception}")
                )
                return failures

            if not response.ok:
                try:
                    details = response.json()
                    error_text = details.get("description", response.text)
                except json.JSONDecodeError:
                    error_text = response.text

                logger.error("Failed to send photo to %s: %s", chat_id, error_text)
                failures.append(
                    DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
                )
            
            return failures

        # Если несколько фото, используем sendMediaGroup
        # Создаем массив InputMediaPhoto
        media_array = []
        files_dict = {}
        
        for idx, photo in enumerate(photos):
            file_id = f"photo_{idx}"
            media_item = {
                "type": "photo",
                "media": f"attach://{file_id}",
            }
            
            # Caption только на первом медиа
            if attach_message and idx == 0:
                media_item["caption"] = message
                if parse_mode:
                    media_item["parse_mode"] = parse_mode
            
            # Reply_markup только на последнем медиа (добавим позже в data)
            # Но в media array его не добавляем, он идет отдельно в data
            
            media_array.append(media_item)
            
            # Добавляем файл в files_dict
            files_dict[file_id] = (
                photo.filename or f"photo_{idx}.jpg",
                photo.content,
                photo.content_type or "image/jpeg",
            )
        
        # Подготавливаем данные для запроса
        data = {
            "chat_id": chat_id,
            "media": json.dumps(media_array, separators=(',', ':'), ensure_ascii=False),
        }
        
        # Reply_markup добавляем к последнему медиа через параметр в data
        # Но согласно документации, reply_markup не поддерживается в sendMediaGroup напрямую
        # Нужно отправить отдельным сообщением или использовать другой подход
        # Однако, можно попробовать добавить reply_markup к последнему элементу media_array
        if inline_keyboard and (attach_message or attach_keyboard_to_last):
            try:
                keyboard_dict = {
                    "inline_keyboard": [
                        [button.to_dict() for button in row] for row in inline_keyboard
                    ]
                }
                keyboard_json = json.dumps(keyboard_dict, separators=(',', ':'), ensure_ascii=False)
                json.loads(keyboard_json)  # Валидация
                
                # В sendMediaGroup reply_markup не поддерживается напрямую
                # Но можно добавить его к последнему элементу media_array
                # Однако это не стандартное поведение API
                # Попробуем добавить reply_markup как отдельный параметр (может не работать)
                # Лучше отправить отдельным сообщением после media group
                logger.warning("Inline keyboard with media group: will be sent separately if needed")
                # Пока не добавляем, так как это не поддерживается в sendMediaGroup
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize inline keyboard: {e}")

        # Вычисляем общий размер файлов для определения таймаута
        total_size_mb = sum(len(photo.content) for photo in photos) / (1024 * 1024)
        timeout = 180 if total_size_mb > 10 else 120 if total_size_mb > 5 else 60
        
        # Retry logic with exponential backoff for SSL/connection errors
        max_retries = 3
        retry_delay = 1
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/sendMediaGroup",
                    data=data,
                    files=files_dict,
                    timeout=timeout,
                )
                # If we get a response, break out of retry loop
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.SSLError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        "Attempt %d/%d failed for chat_id %s: %s. Retrying in %d seconds...",
                        attempt + 1, max_retries, chat_id, e, wait_time
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("Timeout or connection error sending media group to %s after %d attempts: %s", chat_id, max_retries, e)
                    failures.append(
                        DeliveryReport(chat_id=chat_id, ok=False, error=f"Timeout or connection error after {max_retries} attempts: {e}")
                    )
                    return failures
            except Exception as e:
                # For other exceptions, don't retry
                logger.error("Unexpected error sending media group to %s: %s", chat_id, e)
                failures.append(
                    DeliveryReport(chat_id=chat_id, ok=False, error=f"Unexpected error: {e}")
                )
                return failures
        else:
            # This executes if the for loop completes without breaking (shouldn't happen, but safety)
            if last_exception:
                logger.error("Failed to send media group to %s after all retries: %s", chat_id, last_exception)
                failures.append(
                    DeliveryReport(chat_id=chat_id, ok=False, error=f"Failed after {max_retries} attempts: {last_exception}")
                )
                return failures

        # Обработка ошибок с retry без parse_mode при ошибках парсинга entities
        if not response.ok:
            try:
                details = response.json()
                error_text = details.get("description", response.text)
                
                # Если ошибка связана с парсингом entities, пробуем отправить без parse_mode
                if "can't parse" in error_text.lower() and "entities" in error_text.lower() and parse_mode:
                    logger.warning("Entity parsing error detected, retrying without parse_mode for chat_id %s", chat_id)
                    # Создаем новый media_array без parse_mode
                    media_array_no_parse = []
                    for idx, photo in enumerate(photos):
                        file_id = f"photo_{idx}"
                        media_item = {
                            "type": "photo",
                            "media": f"attach://{file_id}",
                        }
                        if attach_message and idx == 0:
                            media_item["caption"] = message
                            # Не добавляем parse_mode
                        media_array_no_parse.append(media_item)
                    
                    data_no_parse = {
                        "chat_id": chat_id,
                        "media": json.dumps(media_array_no_parse, separators=(',', ':'), ensure_ascii=False),
                    }
                    
                    # Retry logic for parse_mode retry
                    max_parse_retries = 2
                    parse_retry_delay = 1
                    parse_last_exception = None
                    retry_response = None
                    
                    for parse_attempt in range(max_parse_retries):
                        try:
                            retry_response = self.session.post(
                                f"{self.base_url}/sendMediaGroup",
                                data=data_no_parse,
                                files=files_dict,
                                timeout=timeout,  # Используем тот же таймаут
                            )
                            break
                        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.SSLError) as e:
                            parse_last_exception = e
                            if parse_attempt < max_parse_retries - 1:
                                wait_time = parse_retry_delay * (2 ** parse_attempt)
                                logger.warning(
                                    "Parse retry attempt %d/%d failed for chat_id %s: %s. Retrying in %d seconds...",
                                    parse_attempt + 1, max_parse_retries, chat_id, e, wait_time
                                )
                                time.sleep(wait_time)
                            else:
                                logger.error("Timeout or connection error on parse retry for chat_id %s: %s", chat_id, e)
                                failures.append(
                                    DeliveryReport(chat_id=chat_id, ok=False, error=f"Timeout or connection error: {e}")
                                )
                                return failures
                    
                    if parse_last_exception and retry_response is None:
                        failures.append(
                            DeliveryReport(chat_id=chat_id, ok=False, error=f"Timeout or connection error: {parse_last_exception}")
                        )
                        return failures
                    
                    if retry_response and retry_response.ok:
                        logger.info("Successfully sent media group without parse_mode for chat_id %s", chat_id)
                        # Продолжаем обработку inline_keyboard как обычно
                        response = retry_response
                    else:
                        # Если и без parse_mode не получилось, возвращаем исходную ошибку
                        logger.error("Failed to send media group even without parse_mode to %s: %s", chat_id, retry_response.text)
                        failures.append(
                            DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
                        )
                        return failures
                else:
                    # Обычная ошибка, не связанная с парсингом entities
                    logger.error("Failed to send media group to %s: %s", chat_id, error_text)
                    failures.append(
                        DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
                    )
                    return failures
            except json.JSONDecodeError:
                error_text = response.text
                logger.error("Failed to send media group to %s: %s", chat_id, error_text)
                failures.append(
                    DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
                )
                return failures

        # Обработка inline_keyboard (выполняется как при успешной отправке, так и после успешного retry)
        if response.ok:
            # Если есть inline_keyboard, добавляем его к последнему сообщению в media group
            if inline_keyboard and (attach_keyboard_to_last or (attach_message and len(photos) > 1)):
                try:
                    # Получаем ответ от sendMediaGroup, чтобы узнать message_id последнего сообщения
                    response_data = response.json()
                    if response_data.get("ok") and response_data.get("result"):
                        messages = response_data["result"]
                        if messages and len(messages) > 0:
                            last_message_id = messages[-1].get("message_id")
                            
                            keyboard_dict = {
                                "inline_keyboard": [
                                    [button.to_dict() for button in row] for row in inline_keyboard
                                ]
                            }
                            
                            # Редактируем последнее сообщение, добавляя reply_markup
                            edit_data = {
                                "chat_id": chat_id,
                                "message_id": last_message_id,
                                "reply_markup": keyboard_dict,
                            }
                            
                            edit_response = self.session.post(
                                f"{self.base_url}/editMessageReplyMarkup",
                                json=edit_data,
                                timeout=15,
                            )
                            
                            if not edit_response.ok:
                                logger.warning("Failed to add inline keyboard to last message in media group")
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    logger.error(f"Error adding inline keyboard to media group: {e}")
                    # Fallback: отправляем отдельным сообщением
                    try:
                        keyboard_dict = {
                            "inline_keyboard": [
                                [button.to_dict() for button in row] for row in inline_keyboard
                            ]
                        }
                        message_data = {
                            "chat_id": chat_id,
                            "text": " ",  # Минимальный текст
                            "reply_markup": keyboard_dict,
                        }
                        self.session.post(
                            f"{self.base_url}/sendMessage",
                            json=message_data,
                            timeout=15,
                        )
                    except Exception as fallback_error:
                        logger.error(f"Fallback keyboard send also failed: {fallback_error}")

        return failures

    def _send_videos(
        self,
        chat_id: Union[int, str],
        videos: Sequence[TelegramVideo],
        attach_message: bool,
        message: str,
        parse_mode: Optional[str],
        inline_keyboard: Optional[List[List[InlineButton]]],
        attach_keyboard_to_last: bool = False,
    ) -> List[DeliveryReport]:
        failures: List[DeliveryReport] = []

        if not videos:
            return failures

        # Если одно видео, используем sendVideo
        if len(videos) == 1:
            video = videos[0]
            data = {"chat_id": chat_id}
            
            if attach_message:
                data["caption"] = message
                if parse_mode:
                    data["parse_mode"] = parse_mode
            
            if video.duration:
                data["duration"] = video.duration
            if video.width:
                data["width"] = video.width
            if video.height:
                data["height"] = video.height
            
            if inline_keyboard:
                try:
                    keyboard_dict = {
                        "inline_keyboard": [
                            [button.to_dict() for button in row] for row in inline_keyboard
                        ]
                    }
                    keyboard_json = json.dumps(keyboard_dict, separators=(',', ':'), ensure_ascii=False)
                    json.loads(keyboard_json)  # Валидация
                    logger.debug(f"Sending reply_markup JSON: {keyboard_json[:200]}...")
                    data["reply_markup"] = keyboard_json
                except (TypeError, ValueError) as e:
                    logger.error(f"Failed to serialize inline keyboard: {e}")

            files = {
                "video": (
                    video.filename or "video.mp4",
                    video.content,
                    video.content_type or "video/mp4",
                )
            }
            
            # Добавляем thumbnail, если есть
            if video.thumbnail:
                files["thumbnail"] = (
                    video.thumbnail.filename or "thumbnail.jpg",
                    video.thumbnail.content,
                    video.thumbnail.content_type or "image/jpeg",
                )

            # Увеличиваем таймаут для больших видео
            video_size_mb = len(video.content) / (1024 * 1024)
            timeout = 300 if video_size_mb > 50 else 180 if video_size_mb > 20 else 120
            
            # Retry logic with exponential backoff for SSL/connection errors
            max_retries = 3
            retry_delay = 1
            last_exception = None
            response = None
            
            for attempt in range(max_retries):
                try:
                    response = self.session.post(
                        f"{self.base_url}/sendVideo",
                        data=data,
                        files=files,
                        timeout=timeout,
                    )
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.SSLError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(
                            "Video send attempt %d/%d failed for chat_id %s: %s. Retrying in %d seconds...",
                            attempt + 1, max_retries, chat_id, e, wait_time
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error("Timeout or connection error sending video to %s after %d attempts: %s", chat_id, max_retries, e)
                        failures.append(
                            DeliveryReport(chat_id=chat_id, ok=False, error=f"Timeout or connection error after {max_retries} attempts: {e}")
                        )
                        return failures
            
            if last_exception and response is None:
                failures.append(
                    DeliveryReport(chat_id=chat_id, ok=False, error=f"Timeout or connection error: {last_exception}")
                )
                return failures

            if not response.ok:
                try:
                    details = response.json()
                    error_text = details.get("description", response.text)
                except json.JSONDecodeError:
                    error_text = response.text

                logger.error("Failed to send video to %s: %s", chat_id, error_text)
                failures.append(
                    DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
                )
            
            return failures

        # Если несколько видео, используем sendMediaGroup
        # Создаем массив InputMediaVideo
        media_array = []
        files_dict = {}
        
        for idx, video in enumerate(videos):
            file_id = f"video_{idx}"
            media_item = {
                "type": "video",
                "media": f"attach://{file_id}",
            }
            
            # Caption только на первом медиа
            if attach_message and idx == 0:
                media_item["caption"] = message
                if parse_mode:
                    media_item["parse_mode"] = parse_mode
            
            # Опциональные параметры видео
            if video.duration:
                media_item["duration"] = video.duration
            if video.width:
                media_item["width"] = video.width
            if video.height:
                media_item["height"] = video.height
            
            media_array.append(media_item)
            
            # Добавляем файл в files_dict
            files_dict[file_id] = (
                video.filename or f"video_{idx}.mp4",
                video.content,
                video.content_type or "video/mp4",
            )
            
            # Добавляем thumbnail, если есть
            if video.thumbnail:
                thumb_id = f"thumb_{idx}"
                files_dict[thumb_id] = (
                    video.thumbnail.filename or f"thumb_{idx}.jpg",
                    video.thumbnail.content,
                    video.thumbnail.content_type or "image/jpeg",
                )
                media_item["thumbnail"] = f"attach://{thumb_id}"
        
        # Подготавливаем данные для запроса
        data = {
            "chat_id": chat_id,
            "media": json.dumps(media_array, separators=(',', ':'), ensure_ascii=False),
        }
        
        if inline_keyboard and (attach_message or attach_keyboard_to_last):
            logger.warning("Inline keyboard with media group: will be sent separately if needed")

        # Вычисляем общий размер файлов для определения таймаута
        total_size_mb = sum(len(video.content) for video in videos) / (1024 * 1024)
        timeout = 300 if total_size_mb > 50 else 180 if total_size_mb > 20 else 120
        
        # Retry logic with exponential backoff for SSL/connection errors
        max_retries = 3
        retry_delay = 1
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/sendMediaGroup",
                    data=data,
                    files=files_dict,
                    timeout=timeout,
                )
                # If we get a response, break out of retry loop
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.SSLError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        "Attempt %d/%d failed for chat_id %s: %s. Retrying in %d seconds...",
                        attempt + 1, max_retries, chat_id, e, wait_time
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("Timeout or connection error sending media group to %s after %d attempts: %s", chat_id, max_retries, e)
                    failures.append(
                        DeliveryReport(chat_id=chat_id, ok=False, error=f"Timeout or connection error after {max_retries} attempts: {e}")
                    )
                    return failures
            except Exception as e:
                # For other exceptions, don't retry
                logger.error("Unexpected error sending media group to %s: %s", chat_id, e)
                failures.append(
                    DeliveryReport(chat_id=chat_id, ok=False, error=f"Unexpected error: {e}")
                )
                return failures
        else:
            # This executes if the for loop completes without breaking (shouldn't happen, but safety)
            if last_exception:
                logger.error("Failed to send media group to %s after all retries: %s", chat_id, last_exception)
                failures.append(
                    DeliveryReport(chat_id=chat_id, ok=False, error=f"Failed after {max_retries} attempts: {last_exception}")
                )
                return failures

        # Обработка ошибок с retry без parse_mode при ошибках парсинга entities
        if not response.ok:
            try:
                details = response.json()
                error_text = details.get("description", response.text)
                
                # Если ошибка связана с парсингом entities, пробуем отправить без parse_mode
                if "can't parse" in error_text.lower() and "entities" in error_text.lower() and parse_mode:
                    logger.warning("Entity parsing error detected, retrying without parse_mode for chat_id %s", chat_id)
                    # Создаем новый media_array без parse_mode
                    media_array_no_parse = []
                    for idx, video in enumerate(videos):
                        file_id = f"video_{idx}"
                        media_item = {
                            "type": "video",
                            "media": f"attach://{file_id}",
                        }
                        if attach_message and idx == 0:
                            media_item["caption"] = message
                            # Не добавляем parse_mode
                        
                        # Опциональные параметры видео
                        if video.duration:
                            media_item["duration"] = video.duration
                        if video.width:
                            media_item["width"] = video.width
                        if video.height:
                            media_item["height"] = video.height
                        
                        media_array_no_parse.append(media_item)
                    
                    data_no_parse = {
                        "chat_id": chat_id,
                        "media": json.dumps(media_array_no_parse, separators=(',', ':'), ensure_ascii=False),
                    }
                    
                    # Retry logic for parse_mode retry
                    max_parse_retries = 2
                    parse_retry_delay = 1
                    parse_last_exception = None
                    retry_response = None
                    
                    for parse_attempt in range(max_parse_retries):
                        try:
                            retry_response = self.session.post(
                                f"{self.base_url}/sendMediaGroup",
                                data=data_no_parse,
                                files=files_dict,
                                timeout=timeout,  # Используем тот же таймаут
                            )
                            break
                        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.SSLError) as e:
                            parse_last_exception = e
                            if parse_attempt < max_parse_retries - 1:
                                wait_time = parse_retry_delay * (2 ** parse_attempt)
                                logger.warning(
                                    "Parse retry attempt %d/%d failed for chat_id %s: %s. Retrying in %d seconds...",
                                    parse_attempt + 1, max_parse_retries, chat_id, e, wait_time
                                )
                                time.sleep(wait_time)
                            else:
                                logger.error("Timeout or connection error on parse retry for chat_id %s: %s", chat_id, e)
                                failures.append(
                                    DeliveryReport(chat_id=chat_id, ok=False, error=f"Timeout or connection error: {e}")
                                )
                                return failures
                    
                    if parse_last_exception and retry_response is None:
                        failures.append(
                            DeliveryReport(chat_id=chat_id, ok=False, error=f"Timeout or connection error: {parse_last_exception}")
                        )
                        return failures
                    
                    if retry_response and retry_response.ok:
                        logger.info("Successfully sent media group without parse_mode for chat_id %s", chat_id)
                        # Продолжаем обработку inline_keyboard как обычно
                        response = retry_response
                    else:
                        # Если и без parse_mode не получилось, возвращаем исходную ошибку
                        logger.error("Failed to send media group even without parse_mode to %s: %s", chat_id, retry_response.text if retry_response else "No response")
                        failures.append(
                            DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
                        )
                        return failures
                else:
                    # Обычная ошибка, не связанная с парсингом entities
                    logger.error("Failed to send media group to %s: %s", chat_id, error_text)
                    failures.append(
                        DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
                    )
                    return failures
            except json.JSONDecodeError:
                error_text = response.text
                logger.error("Failed to send media group to %s: %s", chat_id, error_text)
                failures.append(
                    DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
                )
                return failures

        # Обработка inline_keyboard (выполняется как при успешной отправке, так и после успешного retry)
        if response.ok:
            # Если есть inline_keyboard, добавляем его к последнему сообщению в media group
            if inline_keyboard and (attach_keyboard_to_last or (attach_message and len(videos) > 1)):
                try:
                    # Получаем ответ от sendMediaGroup, чтобы узнать message_id последнего сообщения
                    response_data = response.json()
                    if response_data.get("ok") and response_data.get("result"):
                        messages = response_data["result"]
                        if messages and len(messages) > 0:
                            last_message_id = messages[-1].get("message_id")
                            
                            keyboard_dict = {
                                "inline_keyboard": [
                                    [button.to_dict() for button in row] for row in inline_keyboard
                                ]
                            }
                            
                            # Редактируем последнее сообщение, добавляя reply_markup
                            edit_data = {
                                "chat_id": chat_id,
                                "message_id": last_message_id,
                                "reply_markup": keyboard_dict,
                            }
                            
                            edit_response = self.session.post(
                                f"{self.base_url}/editMessageReplyMarkup",
                                json=edit_data,
                                timeout=15,
                            )
                            
                            if not edit_response.ok:
                                logger.warning("Failed to add inline keyboard to last message in media group")
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    logger.error(f"Error adding inline keyboard to media group: {e}")
                    # Fallback: отправляем отдельным сообщением
                    try:
                        keyboard_dict = {
                            "inline_keyboard": [
                                [button.to_dict() for button in row] for row in inline_keyboard
                            ]
                        }
                        message_data = {
                            "chat_id": chat_id,
                            "text": " ",  # Минимальный текст
                            "reply_markup": keyboard_dict,
                        }
                        self.session.post(
                            f"{self.base_url}/sendMessage",
                            json=message_data,
                            timeout=15,
                        )
                    except Exception as fallback_error:
                        logger.error(f"Fallback keyboard send also failed: {fallback_error}")

        return failures


    async def _send_message_async(
        self,
        chat_id: Union[int, str],
        config: TelegramBroadcastConfig,
    ) -> DeliveryReport:
        """Асинхронная версия _send_message"""
        if not hasattr(self, '_async_session') or self._async_session is None:
            raise RuntimeError("Async session not initialized. Use broadcast_async() method.")
        
        payload = {
            "chat_id": chat_id,
            "text": config.message,
            "disable_web_page_preview": config.disable_web_page_preview,
        }

        if config.parse_mode:
            payload["parse_mode"] = config.parse_mode

        if config.inline_keyboard:
            payload["reply_markup"] = {
                "inline_keyboard": [
                    [button.to_dict() for button in row] for row in config.inline_keyboard
                ]
            }

        if config.extra_api_params:
            payload.update(config.extra_api_params)

        # Retry logic для сетевых ошибок
        max_retries = 3
        retry_delay = 0.5  # Уменьшено с 1 до 0.5 для более быстрой обработки
        
        for attempt in range(max_retries):
            try:
                async with self._async_session.post(
                    f"{self.base_url}/sendMessage",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as response:
                    if response.ok:
                        return DeliveryReport(chat_id=chat_id, ok=True)

                    try:
                        details = await response.json()
                        error_text = details.get("description", await response.text())
                    except Exception:
                        error_text = await response.text()

                    # Для ошибок типа "chat not found" не делаем retry
                    if "chat not found" in error_text.lower() or "chat_id" in error_text.lower():
                        logger.warning("Failed to send message to %s: %s (no retry)", chat_id, error_text)
                        return DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
                    
                    # Если ошибка связана с парсингом entities, пробуем отправить без parse_mode
                    if "can't parse" in error_text.lower() and "entities" in error_text.lower() and config.parse_mode:
                        logger.warning("Entity parsing error detected, retrying without parse_mode for chat_id %s", chat_id)
                        # Retry without parse_mode
                        retry_payload = {
                            "chat_id": chat_id,
                            "text": config.message,
                            "disable_web_page_preview": config.disable_web_page_preview,
                        }
                        # Не добавляем parse_mode
                        
                        if config.inline_keyboard:
                            retry_payload["reply_markup"] = {
                                "inline_keyboard": [
                                    [button.to_dict() for button in row] for row in config.inline_keyboard
                                ]
                            }
                        
                        if config.extra_api_params:
                            retry_payload.update(config.extra_api_params)
                        
                        try:
                            async with self._async_session.post(
                                f"{self.base_url}/sendMessage",
                                json=retry_payload,
                                timeout=aiohttp.ClientTimeout(total=15),
                            ) as retry_response:
                                if retry_response.ok:
                                    logger.info("Successfully sent message without parse_mode for chat_id %s", chat_id)
                                    return DeliveryReport(chat_id=chat_id, ok=True)
                                else:
                                    # Если и без parse_mode не получилось, возвращаем исходную ошибку
                                    try:
                                        retry_details = await retry_response.json()
                                        retry_error_text = retry_details.get("description", await retry_response.text())
                                    except Exception:
                                        retry_error_text = await retry_response.text()
                                    logger.error("Failed to send message even without parse_mode to %s: %s", chat_id, retry_error_text)
                                    return DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
                        except Exception as retry_e:
                            logger.error("Error during parse_mode retry for message to %s: %s", chat_id, retry_e)
                            return DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
                    
                    # Для других ошибок API не делаем retry
                    logger.error("Failed to send message to %s: %s", chat_id, error_text)
                    return DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
            except (asyncio.TimeoutError, aiohttp.ClientError, aiohttp.ServerConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        "Connection error sending message to %s (attempt %d/%d): %s. Retrying in %d seconds...",
                        chat_id, attempt + 1, max_retries, e, wait_time
                    )
                    await asyncio.sleep(wait_time)
                else:
                    error_text = f"Connection error after {max_retries} attempts: {e}"
                    logger.error("Connection error sending message to %s after %d attempts: %s", chat_id, max_retries, e)
                    return DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
            except Exception as e:
                error_text = f"Unexpected error: {e}"
                logger.error("Unexpected error sending message to %s: %s", chat_id, e, exc_info=True)
                return DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
        
        # Fallback (не должно достигнуться)
        return DeliveryReport(chat_id=chat_id, ok=False, error="Failed after all retries")
    async def _send_photos_async(
        self,
        chat_id: Union[int, str],
        photos: Sequence[TelegramPhoto],
        attach_message: bool,
        message: str,
        parse_mode: Optional[str],
        inline_keyboard: Optional[List[List[InlineButton]]],
        attach_keyboard_to_last: bool = False,
    ) -> List[DeliveryReport]:
        """Асинхронная версия _send_photos"""
        failures: List[DeliveryReport] = []

        if not photos:
            return failures

        if not hasattr(self, '_async_session') or self._async_session is None:
            raise RuntimeError("Async session not initialized. Use broadcast_async() method.")

        if len(photos) == 1:
            photo = photos[0]
            timeout = 120 if len(photo.content) / (1024 * 1024) > 5 else 60 if len(photo.content) / (1024 * 1024) > 1 else 30
            
            # Retry logic для сетевых ошибок
            max_retries = 3
            retry_delay = 1
            success = False
            
            for attempt in range(max_retries):
                # Recreate FormData for each retry attempt (FormData can only be processed once)
                data = aiohttp.FormData()
                data.add_field("chat_id", str(chat_id))
                
                if attach_message:
                    data.add_field("caption", message)
                    if parse_mode:
                        data.add_field("parse_mode", parse_mode)
                
                if inline_keyboard:
                    try:
                        keyboard_dict = {
                            "inline_keyboard": [
                                [button.to_dict() for button in row] for row in inline_keyboard
                            ]
                        }
                        keyboard_json = json.dumps(keyboard_dict, separators=(',', ':'), ensure_ascii=False)
                        data.add_field("reply_markup", keyboard_json)
                    except (TypeError, ValueError) as e:
                        logger.error(f"Failed to serialize inline keyboard: {e}")

                data.add_field("photo", photo.content, filename=photo.filename or "photo.jpg", content_type=photo.content_type or "image/jpeg")
                
                try:
                    async with self._async_session.post(
                        f"{self.base_url}/sendPhoto",
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as response:
                        if response.ok:
                            success = True
                            break
                        
                        try:
                            details = await response.json()
                            error_text = details.get("description", await response.text())
                        except Exception:
                            error_text = await response.text()
                        
                        # Для ошибок типа "chat not found" не делаем retry
                        if "chat not found" in error_text.lower() or "chat_id" in error_text.lower():
                            logger.warning("Failed to send photo to %s: %s (no retry)", chat_id, error_text)
                            failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                            return failures
                        
                        # Если ошибка связана с парсингом entities, пробуем отправить без parse_mode
                        if "can't parse" in error_text.lower() and "entities" in error_text.lower() and parse_mode:
                            logger.warning("Entity parsing error detected, retrying without parse_mode for chat_id %s", chat_id)
                            # Retry without parse_mode
                            retry_data = aiohttp.FormData()
                            retry_data.add_field("chat_id", str(chat_id))
                            
                            if attach_message:
                                retry_data.add_field("caption", message)
                                # Не добавляем parse_mode
                            
                            if inline_keyboard:
                                try:
                                    keyboard_dict = {
                                        "inline_keyboard": [
                                            [button.to_dict() for button in row] for row in inline_keyboard
                                        ]
                                    }
                                    keyboard_json = json.dumps(keyboard_dict, separators=(',', ':'), ensure_ascii=False)
                                    retry_data.add_field("reply_markup", keyboard_json)
                                except (TypeError, ValueError) as e:
                                    logger.error(f"Failed to serialize inline keyboard: {e}")
                            
                            retry_data.add_field("photo", photo.content, filename=photo.filename or "photo.jpg", content_type=photo.content_type or "image/jpeg")
                            
                            try:
                                async with self._async_session.post(
                                    f"{self.base_url}/sendPhoto",
                                    data=retry_data,
                                    timeout=aiohttp.ClientTimeout(total=timeout),
                                ) as retry_response:
                                    if retry_response.ok:
                                        logger.info("Successfully sent photo without parse_mode for chat_id %s", chat_id)
                                        return failures  # Success
                                    else:
                                        # Если и без parse_mode не получилось, возвращаем исходную ошибку
                                        try:
                                            retry_details = await retry_response.json()
                                            retry_error_text = retry_details.get("description", await retry_response.text())
                                        except Exception:
                                            retry_error_text = await retry_response.text()
                                        logger.error("Failed to send photo even without parse_mode to %s: %s", chat_id, retry_error_text)
                                        failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                                        return failures
                            except Exception as retry_e:
                                logger.error("Error during parse_mode retry for photo to %s: %s", chat_id, retry_e)
                                failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                                return failures
                        else:
                            # Для других ошибок API не делаем retry
                            logger.error("Failed to send photo to %s: %s", chat_id, error_text)
                            failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                            return failures
                except (asyncio.TimeoutError, aiohttp.ClientError, aiohttp.ServerConnectionError) as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(
                            "Connection error sending photo to %s (attempt %d/%d): %s. Retrying in %d seconds...",
                            chat_id, attempt + 1, max_retries, e, wait_time
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("Connection error sending photo to %s after %d attempts: %s", chat_id, max_retries, e)
                        failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=f"Connection error after {max_retries} attempts: {e}"))
                        return failures
                except Exception as e:
                    logger.error("Unexpected error sending photo to %s: %s", chat_id, e, exc_info=True)
                    failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=f"Unexpected error: {e}"))
                    return failures
            
            if not success:
                failures.append(DeliveryReport(chat_id=chat_id, ok=False, error="Failed after all retries"))
            
            return failures

        # Для нескольких фото используем sendMediaGroup
        total_size_mb = sum(len(photo.content) for photo in photos) / (1024 * 1024)
        timeout = 180 if total_size_mb > 10 else 120 if total_size_mb > 5 else 60
        
        # Retry logic для сетевых ошибок
        max_retries = 3
        retry_delay = 0.5  # Уменьшено с 1 до 0.5 для более быстрой обработки
        success = False
        
        for attempt in range(max_retries):
            # Recreate FormData for each retry attempt (FormData can only be processed once)
            media_array = []
            data = aiohttp.FormData()
            data.add_field("chat_id", str(chat_id))
            
            for idx, photo in enumerate(photos):
                file_id = f"photo_{idx}"
                media_item = {
                    "type": "photo",
                    "media": f"attach://{file_id}",
                }
                
                if attach_message and idx == 0:
                    media_item["caption"] = message
                    if parse_mode:
                        media_item["parse_mode"] = parse_mode
                
                media_array.append(media_item)
                data.add_field(file_id, photo.content, filename=photo.filename or f"photo_{idx}.jpg", content_type=photo.content_type or "image/jpeg")
            
            data.add_field("media", json.dumps(media_array, separators=(',', ':'), ensure_ascii=False))
            
            try:
                async with self._async_session.post(
                    f"{self.base_url}/sendMediaGroup",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    if not response.ok:
                        try:
                            details = await response.json()
                            error_text = details.get("description", await response.text())
                        except Exception:
                            error_text = await response.text()
                        
                        # Для ошибок типа "chat not found" не делаем retry
                        if "chat not found" in error_text.lower() or "chat_id" in error_text.lower():
                            logger.warning("Failed to send media group to %s: %s (no retry)", chat_id, error_text)
                            failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                            return failures
                        
                        # Если ошибка связана с парсингом entities, пробуем отправить без parse_mode
                        if "can't parse" in error_text.lower() and "entities" in error_text.lower() and parse_mode:
                            logger.warning("Entity parsing error detected, retrying without parse_mode for chat_id %s", chat_id)
                            # Retry without parse_mode
                            retry_media_array = []
                            retry_data = aiohttp.FormData()
                            retry_data.add_field("chat_id", str(chat_id))
                            
                            for idx, photo in enumerate(photos):
                                file_id = f"photo_{idx}"
                                media_item = {
                                    "type": "photo",
                                    "media": f"attach://{file_id}",
                                }
                                
                                if attach_message and idx == 0:
                                    media_item["caption"] = message
                                    # Не добавляем parse_mode
                                
                                retry_media_array.append(media_item)
                                retry_data.add_field(file_id, photo.content, filename=photo.filename or f"photo_{idx}.jpg", content_type=photo.content_type or "image/jpeg")
                            
                            retry_data.add_field("media", json.dumps(retry_media_array, separators=(',', ':'), ensure_ascii=False))
                            
                            try:
                                async with self._async_session.post(
                                    f"{self.base_url}/sendMediaGroup",
                                    data=retry_data,
                                    timeout=aiohttp.ClientTimeout(total=timeout),
                                ) as retry_response:
                                    if retry_response.ok:
                                        logger.info("Successfully sent media group without parse_mode for chat_id %s", chat_id)
                                        # Обработка inline keyboard
                                        if inline_keyboard and (attach_keyboard_to_last or (attach_message and len(photos) > 1)):
                                            try:
                                                retry_response_data = await retry_response.json()
                                                if retry_response_data.get("ok") and retry_response_data.get("result"):
                                                    messages = retry_response_data["result"]
                                                    if messages and len(messages) > 0:
                                                        last_message_id = messages[-1].get("message_id")
                                                        keyboard_dict = {
                                                            "inline_keyboard": [
                                                                [button.to_dict() for button in row] for row in inline_keyboard
                                                            ]
                                                        }
                                                        edit_data = {
                                                            "chat_id": chat_id,
                                                            "message_id": last_message_id,
                                                            "reply_markup": keyboard_dict,
                                                        }
                                                        async with self._async_session.post(
                                                            f"{self.base_url}/editMessageReplyMarkup",
                                                            json=edit_data,
                                                            timeout=aiohttp.ClientTimeout(total=15),
                                                        ) as edit_response:
                                                            if not edit_response.ok:
                                                                logger.warning("Failed to add inline keyboard to last message in media group")
                                            except Exception as e:
                                                logger.error(f"Error adding inline keyboard to media group: {e}")
                                        return failures  # Success
                                    else:
                                        # Если и без parse_mode не получилось, возвращаем исходную ошибку
                                        try:
                                            retry_details = await retry_response.json()
                                            retry_error_text = retry_details.get("description", await retry_response.text())
                                        except Exception:
                                            retry_error_text = await retry_response.text()
                                        logger.error("Failed to send media group even without parse_mode to %s: %s", chat_id, retry_error_text)
                                        failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                                        return failures
                            except Exception as retry_e:
                                logger.error("Error during parse_mode retry for media group to %s: %s", chat_id, retry_e)
                                failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                                return failures
                        else:
                            # Для других ошибок API не делаем retry
                            logger.error("Failed to send media group to %s: %s", chat_id, error_text)
                            failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                            return failures
                    
                    success = True
                    # Обработка inline keyboard
                    if inline_keyboard and (attach_keyboard_to_last or (attach_message and len(photos) > 1)):
                        try:
                            response_data = await response.json()
                            if response_data.get("ok") and response_data.get("result"):
                                messages = response_data["result"]
                                if messages and len(messages) > 0:
                                    last_message_id = messages[-1].get("message_id")
                                    keyboard_dict = {
                                        "inline_keyboard": [
                                            [button.to_dict() for button in row] for row in inline_keyboard
                                        ]
                                    }
                                    edit_data = {
                                        "chat_id": chat_id,
                                        "message_id": last_message_id,
                                        "reply_markup": keyboard_dict,
                                    }
                                    async with self._async_session.post(
                                        f"{self.base_url}/editMessageReplyMarkup",
                                        json=edit_data,
                                        timeout=aiohttp.ClientTimeout(total=15),
                                    ) as edit_response:
                                        if not edit_response.ok:
                                            logger.warning("Failed to add inline keyboard to last message in media group")
                        except Exception as e:
                            logger.error(f"Error adding inline keyboard to media group: {e}")
                    break  # Успешно отправили, выходим из цикла retry
            except (asyncio.TimeoutError, aiohttp.ClientError, aiohttp.ServerConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        "Connection error sending media group to %s (attempt %d/%d): %s. Retrying in %d seconds...",
                        chat_id, attempt + 1, max_retries, e, wait_time
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Connection error sending media group to %s after %d attempts: %s", chat_id, max_retries, e)
                    failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=f"Connection error after {max_retries} attempts: {e}"))
                    return failures
            except Exception as e:
                logger.error("Unexpected error sending media group to %s: %s", chat_id, e, exc_info=True)
                failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=f"Unexpected error: {e}"))
                return failures
        
        if not success:
            failures.append(DeliveryReport(chat_id=chat_id, ok=False, error="Failed after all retries"))
        
        return failures

    async def _send_videos_async(
        self,
        chat_id: Union[int, str],
        videos: Sequence[TelegramVideo],
        attach_message: bool,
        message: str,
        parse_mode: Optional[str],
        inline_keyboard: Optional[List[List[InlineButton]]],
        attach_keyboard_to_last: bool = False,
    ) -> List[DeliveryReport]:
        """Асинхронная версия _send_videos"""
        failures: List[DeliveryReport] = []

        if not videos:
            return failures

        if not hasattr(self, '_async_session') or self._async_session is None:
            raise RuntimeError("Async session not initialized. Use broadcast_async() method.")

        if len(videos) == 1:
            video = videos[0]
            video_size_mb = len(video.content) / (1024 * 1024)
            timeout = 300 if video_size_mb > 50 else 180 if video_size_mb > 20 else 120
            
            # Retry logic для сетевых ошибок
            max_retries = 3
            retry_delay = 1
            success = False
            
            for attempt in range(max_retries):
                # Recreate FormData for each retry attempt (FormData can only be processed once)
                data = aiohttp.FormData()
                data.add_field("chat_id", str(chat_id))
                
                if attach_message:
                    data.add_field("caption", message)
                    if parse_mode:
                        data.add_field("parse_mode", parse_mode)
                
                if video.duration:
                    data.add_field("duration", str(video.duration))
                if video.width:
                    data.add_field("width", str(video.width))
                if video.height:
                    data.add_field("height", str(video.height))
                
                if inline_keyboard:
                    try:
                        keyboard_dict = {
                            "inline_keyboard": [
                                [button.to_dict() for button in row] for row in inline_keyboard
                            ]
                        }
                        keyboard_json = json.dumps(keyboard_dict, separators=(',', ':'), ensure_ascii=False)
                        data.add_field("reply_markup", keyboard_json)
                    except (TypeError, ValueError) as e:
                        logger.error(f"Failed to serialize inline keyboard: {e}")

                data.add_field("video", video.content, filename=video.filename or "video.mp4", content_type=video.content_type or "video/mp4")
                
                if video.thumbnail:
                    data.add_field("thumbnail", video.thumbnail.content, filename=video.thumbnail.filename or "thumbnail.jpg", content_type=video.thumbnail.content_type or "image/jpeg")
                
                try:
                    async with self._async_session.post(
                        f"{self.base_url}/sendVideo",
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as response:
                        if response.ok:
                            success = True
                            break
                        
                        try:
                            details = await response.json()
                            error_text = details.get("description", await response.text())
                        except Exception:
                            error_text = await response.text()
                        
                        # Для ошибок типа "chat not found" не делаем retry
                        if "chat not found" in error_text.lower() or "chat_id" in error_text.lower():
                            logger.warning("Failed to send video to %s: %s (no retry)", chat_id, error_text)
                            failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                            return failures
                        
                        # Если ошибка связана с парсингом entities, пробуем отправить без parse_mode
                        if "can't parse" in error_text.lower() and "entities" in error_text.lower() and parse_mode:
                            logger.warning("Entity parsing error detected, retrying without parse_mode for chat_id %s", chat_id)
                            # Retry without parse_mode
                            retry_data = aiohttp.FormData()
                            retry_data.add_field("chat_id", str(chat_id))
                            
                            if attach_message:
                                retry_data.add_field("caption", message)
                                # Не добавляем parse_mode
                            
                            if video.duration:
                                retry_data.add_field("duration", str(video.duration))
                            if video.width:
                                retry_data.add_field("width", str(video.width))
                            if video.height:
                                retry_data.add_field("height", str(video.height))
                            
                            if inline_keyboard:
                                try:
                                    keyboard_dict = {
                                        "inline_keyboard": [
                                            [button.to_dict() for button in row] for row in inline_keyboard
                                        ]
                                    }
                                    keyboard_json = json.dumps(keyboard_dict, separators=(',', ':'), ensure_ascii=False)
                                    retry_data.add_field("reply_markup", keyboard_json)
                                except (TypeError, ValueError) as e:
                                    logger.error(f"Failed to serialize inline keyboard: {e}")
                            
                            retry_data.add_field("video", video.content, filename=video.filename or "video.mp4", content_type=video.content_type or "video/mp4")
                            
                            if video.thumbnail:
                                retry_data.add_field("thumbnail", video.thumbnail.content, filename=video.thumbnail.filename or "thumbnail.jpg", content_type=video.thumbnail.content_type or "image/jpeg")
                            
                            try:
                                async with self._async_session.post(
                                    f"{self.base_url}/sendVideo",
                                    data=retry_data,
                                    timeout=aiohttp.ClientTimeout(total=timeout),
                                ) as retry_response:
                                    if retry_response.ok:
                                        logger.info("Successfully sent video without parse_mode for chat_id %s", chat_id)
                                        return failures  # Success
                                    else:
                                        # Если и без parse_mode не получилось, возвращаем исходную ошибку
                                        try:
                                            retry_details = await retry_response.json()
                                            retry_error_text = retry_details.get("description", await retry_response.text())
                                        except Exception:
                                            retry_error_text = await retry_response.text()
                                        logger.error("Failed to send video even without parse_mode to %s: %s", chat_id, retry_error_text)
                                        failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                                        return failures
                            except Exception as retry_e:
                                logger.error("Error during parse_mode retry for video to %s: %s", chat_id, retry_e)
                                failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                                return failures
                        else:
                            # Для других ошибок API не делаем retry
                            logger.error("Failed to send video to %s: %s", chat_id, error_text)
                            failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                            return failures
                except (asyncio.TimeoutError, aiohttp.ClientError, aiohttp.ServerConnectionError) as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(
                            "Connection error sending video to %s (attempt %d/%d): %s. Retrying in %d seconds...",
                            chat_id, attempt + 1, max_retries, e, wait_time
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("Connection error sending video to %s after %d attempts: %s", chat_id, max_retries, e)
                        failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=f"Connection error after {max_retries} attempts: {e}"))
                        return failures
                except Exception as e:
                    logger.error("Unexpected error sending video to %s: %s", chat_id, e, exc_info=True)
                    failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=f"Unexpected error: {e}"))
                    return failures
            
            if not success:
                failures.append(DeliveryReport(chat_id=chat_id, ok=False, error="Failed after all retries"))
            
            return failures

        # Для нескольких видео используем sendMediaGroup
        total_size_mb = sum(len(video.content) for video in videos) / (1024 * 1024)
        timeout = 300 if total_size_mb > 50 else 180 if total_size_mb > 20 else 120
        
        # Retry logic для сетевых ошибок
        max_retries = 3
        retry_delay = 0.5  # Уменьшено с 1 до 0.5 для более быстрой обработки
        success = False
        
        for attempt in range(max_retries):
            # Recreate FormData for each retry attempt (FormData can only be processed once)
            media_array = []
            data = aiohttp.FormData()
            data.add_field("chat_id", str(chat_id))
            
            for idx, video in enumerate(videos):
                file_id = f"video_{idx}"
                media_item = {
                    "type": "video",
                    "media": f"attach://{file_id}",
                }
                
                if attach_message and idx == 0:
                    media_item["caption"] = message
                    if parse_mode:
                        media_item["parse_mode"] = parse_mode
                
                if video.duration:
                    media_item["duration"] = video.duration
                if video.width:
                    media_item["width"] = video.width
                if video.height:
                    media_item["height"] = video.height
                
                media_array.append(media_item)
                data.add_field(file_id, video.content, filename=video.filename or f"video_{idx}.mp4", content_type=video.content_type or "video/mp4")
                
                if video.thumbnail:
                    thumb_id = f"thumb_{idx}"
                    data.add_field(thumb_id, video.thumbnail.content, filename=video.thumbnail.filename or f"thumb_{idx}.jpg", content_type=video.thumbnail.content_type or "image/jpeg")
                    media_item["thumbnail"] = f"attach://{thumb_id}"
            
            data.add_field("media", json.dumps(media_array, separators=(',', ':'), ensure_ascii=False))
            
            try:
                async with self._async_session.post(
                    f"{self.base_url}/sendMediaGroup",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    if not response.ok:
                        try:
                            details = await response.json()
                            error_text = details.get("description", await response.text())
                        except Exception:
                            error_text = await response.text()
                        
                        # Для ошибок типа "chat not found" не делаем retry
                        if "chat not found" in error_text.lower() or "chat_id" in error_text.lower():
                            logger.warning("Failed to send media group to %s: %s (no retry)", chat_id, error_text)
                            failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                            return failures
                        
                        # Если ошибка связана с парсингом entities, пробуем отправить без parse_mode
                        if "can't parse" in error_text.lower() and "entities" in error_text.lower() and parse_mode:
                            logger.warning("Entity parsing error detected, retrying without parse_mode for chat_id %s", chat_id)
                            # Retry without parse_mode
                            retry_media_array = []
                            retry_data = aiohttp.FormData()
                            retry_data.add_field("chat_id", str(chat_id))
                            
                            for idx, video in enumerate(videos):
                                file_id = f"video_{idx}"
                                media_item = {
                                    "type": "video",
                                    "media": f"attach://{file_id}",
                                }
                                
                                if attach_message and idx == 0:
                                    media_item["caption"] = message
                                    # Не добавляем parse_mode
                                
                                if video.duration:
                                    media_item["duration"] = video.duration
                                if video.width:
                                    media_item["width"] = video.width
                                if video.height:
                                    media_item["height"] = video.height
                                
                                retry_media_array.append(media_item)
                                retry_data.add_field(file_id, video.content, filename=video.filename or f"video_{idx}.mp4", content_type=video.content_type or "video/mp4")
                                
                                if video.thumbnail:
                                    thumb_id = f"thumb_{idx}"
                                    retry_data.add_field(thumb_id, video.thumbnail.content, filename=video.thumbnail.filename or f"thumb_{idx}.jpg", content_type=video.thumbnail.content_type or "image/jpeg")
                                    media_item["thumbnail"] = f"attach://{thumb_id}"
                            
                            retry_data.add_field("media", json.dumps(retry_media_array, separators=(',', ':'), ensure_ascii=False))
                            
                            try:
                                async with self._async_session.post(
                                    f"{self.base_url}/sendMediaGroup",
                                    data=retry_data,
                                    timeout=aiohttp.ClientTimeout(total=timeout),
                                ) as retry_response:
                                    if retry_response.ok:
                                        logger.info("Successfully sent media group without parse_mode for chat_id %s", chat_id)
                                        # Обработка inline keyboard
                                        if inline_keyboard and (attach_keyboard_to_last or (attach_message and len(videos) > 1)):
                                            try:
                                                retry_response_data = await retry_response.json()
                                                if retry_response_data.get("ok") and retry_response_data.get("result"):
                                                    messages = retry_response_data["result"]
                                                    if messages and len(messages) > 0:
                                                        last_message_id = messages[-1].get("message_id")
                                                        keyboard_dict = {
                                                            "inline_keyboard": [
                                                                [button.to_dict() for button in row] for row in inline_keyboard
                                                            ]
                                                        }
                                                        edit_data = {
                                                            "chat_id": chat_id,
                                                            "message_id": last_message_id,
                                                            "reply_markup": keyboard_dict,
                                                        }
                                                        async with self._async_session.post(
                                                            f"{self.base_url}/editMessageReplyMarkup",
                                                            json=edit_data,
                                                            timeout=aiohttp.ClientTimeout(total=15),
                                                        ) as edit_response:
                                                            if not edit_response.ok:
                                                                logger.warning("Failed to add inline keyboard to last message in media group")
                                            except Exception as e:
                                                logger.error(f"Error adding inline keyboard to media group: {e}")
                                        return failures  # Success
                                    else:
                                        # Если и без parse_mode не получилось, возвращаем исходную ошибку
                                        try:
                                            retry_details = await retry_response.json()
                                            retry_error_text = retry_details.get("description", await retry_response.text())
                                        except Exception:
                                            retry_error_text = await retry_response.text()
                                        logger.error("Failed to send media group even without parse_mode to %s: %s", chat_id, retry_error_text)
                                        failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                                        return failures
                            except Exception as retry_e:
                                logger.error("Error during parse_mode retry for media group to %s: %s", chat_id, retry_e)
                                failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                                return failures
                        else:
                            # Для других ошибок API не делаем retry
                            logger.error("Failed to send media group to %s: %s", chat_id, error_text)
                            failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=error_text))
                            return failures
                    
                    success = True
                    # Обработка inline keyboard
                    if inline_keyboard and (attach_keyboard_to_last or (attach_message and len(videos) > 1)):
                        try:
                            response_data = await response.json()
                            if response_data.get("ok") and response_data.get("result"):
                                messages = response_data["result"]
                                if messages and len(messages) > 0:
                                    last_message_id = messages[-1].get("message_id")
                                    keyboard_dict = {
                                        "inline_keyboard": [
                                            [button.to_dict() for button in row] for row in inline_keyboard
                                        ]
                                    }
                                    edit_data = {
                                        "chat_id": chat_id,
                                        "message_id": last_message_id,
                                        "reply_markup": keyboard_dict,
                                    }
                                    async with self._async_session.post(
                                        f"{self.base_url}/editMessageReplyMarkup",
                                        json=edit_data,
                                        timeout=aiohttp.ClientTimeout(total=15),
                                    ) as edit_response:
                                        if not edit_response.ok:
                                            logger.warning("Failed to add inline keyboard to last message in media group")
                        except Exception as e:
                            logger.error(f"Error adding inline keyboard to media group: {e}")
                    break  # Успешно отправили, выходим из цикла retry
            except (asyncio.TimeoutError, aiohttp.ClientError, aiohttp.ServerConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        "Connection error sending media group to %s (attempt %d/%d): %s. Retrying in %d seconds...",
                        chat_id, attempt + 1, max_retries, e, wait_time
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Connection error sending media group to %s after %d attempts: %s", chat_id, max_retries, e)
                    failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=f"Connection error after {max_retries} attempts: {e}"))
                    return failures
            except Exception as e:
                logger.error("Unexpected error sending media group to %s: %s", chat_id, e, exc_info=True)
                failures.append(DeliveryReport(chat_id=chat_id, ok=False, error=f"Unexpected error: {e}"))
                return failures
        
        if not success:
            failures.append(DeliveryReport(chat_id=chat_id, ok=False, error="Failed after all retries"))
        
        return failures

