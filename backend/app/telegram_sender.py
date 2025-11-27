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
    attach_message_to_first_photo: bool = False
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

        for chat_id in chat_ids:
            if photos:
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

        if not response.ok:
            try:
                details = response.json()
                error_text = details.get("description", response.text)
            except json.JSONDecodeError:
                error_text = response.text

            logger.error("Failed to send media group to %s: %s", chat_id, error_text)
            failures.append(
                DeliveryReport(chat_id=chat_id, ok=False, error=error_text)
            )
        else:
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

