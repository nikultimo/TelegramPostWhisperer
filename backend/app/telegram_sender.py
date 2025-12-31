from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterable, List, Optional, Sequence, Tuple, Union

import requests

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
) -> List[str]:
    """
    Load chat identifiers from a CSV file-like input.

    The CSV may contain a column named ``telegram_id`` or provide the IDs in the first column.
    Empty rows and malformed entries are skipped.
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

    ids: List[str] = []
    column_idx = (
        header.index(column_name) if header and column_name in header else 0
    )

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
        ids.append(normalized)

    return ids


def _looks_like_header(header: Sequence[str], column_name: str) -> bool:
    if not header:
        return False
    normalized = [cell.lower().strip() for cell in header if cell]
    return column_name.lower() in normalized


class TelegramSender:
    api_host = "https://api.telegram.org"

    def __init__(self, token: str, *, session: Optional[requests.Session] = None):
        self.token = token
        self.session = session or requests.Session()

    @property
    def base_url(self) -> str:
        return f"{self.api_host}/bot{self.token}"

    def broadcast(
        self,
        chat_ids: Iterable[Union[int, str]],
        config: TelegramBroadcastConfig,
    ) -> BroadcastSummary:
        chat_ids = list(chat_ids)
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

            response = self.session.post(
                f"{self.base_url}/sendPhoto",
                data=data,
                files=files,
                timeout=30,
            )

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

        response = self.session.post(
            f"{self.base_url}/sendMediaGroup",
            data=data,
            files=files_dict,
            timeout=30,
        )

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
                    
                    retry_response = self.session.post(
                        f"{self.base_url}/sendMediaGroup",
                        data=data_no_parse,
                        files=files_dict,
                        timeout=30,
                    )
                    
                    if retry_response.ok:
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

            response = self.session.post(
                f"{self.base_url}/sendVideo",
                data=data,
                files=files,
                timeout=60,  # Видео может быть большим, увеличиваем таймаут
            )

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

        response = self.session.post(
            f"{self.base_url}/sendMediaGroup",
            data=data,
            files=files_dict,
            timeout=60,  # Видео может быть большим, увеличиваем таймаут
        )

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
                    
                    retry_response = self.session.post(
                        f"{self.base_url}/sendMediaGroup",
                        data=data_no_parse,
                        files=files_dict,
                        timeout=60,
                    )
                    
                    if retry_response.ok:
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

