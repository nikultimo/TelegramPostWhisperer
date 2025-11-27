#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from backend.app.telegram_sender import (
    InlineButton,
    TelegramBroadcastConfig,
    TelegramPhoto,
    TelegramSender,
    load_chat_ids_from_csv,
)

DEFAULT_MESSAGE = (
    "Привет! Хочу посоветовать тебе пару классных сервисов для общения и подарков.\n\n"
    "1. [Together](https://together.sevostianovs.ru/) — помогает парам планировать общие активности и укреплять отношения. "
    "Загляни и подключи @RelationshipTogetherBot, чтобы напоминания и идеи были всегда под рукой.\n"
    "2. [WishShare](https://wishshare.sevostianovs.ru/) — твой персональный wishlist с ботом @happywishlistbot. "
    "Делись желаниями, собирай подарки и удивляй близких.\n\n"
    "Переходи по ссылкам, попробуй и расскажи, как тебе 👍"
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Рассылка сообщений пользователям Telegram через бота."
    )
    parser.add_argument("--token", required=True, help="Токен Telegram-бота")
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="Путь к CSV файлу с колонкой telegram_id",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--message",
        help="Текст сообщения в формате Markdown",
    )
    group.add_argument(
        "--message-file",
        type=Path,
        help="Путь к файлу с текстом сообщения",
    )
    parser.add_argument(
        "--parse-mode",
        default="Markdown",
        choices=["Markdown", "MarkdownV2", "HTML", "None"],
        help="Режим форматирования Telegram",
    )
    parser.add_argument(
        "--disable-preview",
        action="store_true",
        help="Отключить предпросмотр ссылок",
    )
    parser.add_argument(
        "--inline-keyboard",
        help="JSON строка или путь к JSON файлу с inline-кнопками",
    )
    parser.add_argument(
        "--photo",
        action="append",
        type=Path,
        help="Добавить фото к рассылке (можно указать несколько раз)",
    )
    parser.add_argument(
        "--attach-message-to-first-photo",
        action="store_true",
        help="Отправить текст сообщения как подпись к первой фотографии",
    )
    parser.add_argument(
        "--extra-api-params",
        help="Дополнительные параметры для Telegram API в формате JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать список получателей без отправки сообщений",
    )

    return parser.parse_args(argv)


def load_message(args: argparse.Namespace) -> str:
    if args.message:
        return args.message
    if args.message_file:
        return args.message_file.read_text(encoding="utf-8")
    return DEFAULT_MESSAGE


def load_inline_keyboard(argument: Optional[str]) -> Optional[List[List[InlineButton]]]:
    if not argument:
        return None

    path = Path(argument)
    if path.exists():
        raw = path.read_text(encoding="utf-8")
    else:
        raw = argument

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Ошибка чтения inline-кнопок: {exc}") from exc

    try:
        return [
            [InlineButton(**button_data) for button_data in row]
            for row in parsed
        ]
    except TypeError as exc:
        raise SystemExit(f"Некорректные данные кнопок: {exc}") from exc


def load_photos(arguments: Optional[List[Path]]) -> List[TelegramPhoto]:
    photos: List[TelegramPhoto] = []
    if not arguments:
        return photos

    for path in arguments:
        if not path.exists():
            raise SystemExit(f"Файл {path} не найден")
        photos.append(
            TelegramPhoto(
                filename=path.name,
                content=path.read_bytes(),
                content_type=_guess_mime(path.suffix.lower()),
            )
        )
    return photos


def _guess_mime(suffix: str) -> str:
    return {
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.csv.exists():
        print(f"CSV файл {args.csv} не найден", file=sys.stderr)
        return 1

    try:
        chat_ids = load_chat_ids_from_csv(args.csv)
    except Exception as exc:
        print(f"Не удалось прочитать CSV: {exc}", file=sys.stderr)
        return 1

    if not chat_ids:
        print("В CSV не найдено ни одного telegram_id", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Всего получателей: {len(chat_ids)}")
        print("Примеры ID:", ", ".join(chat_ids[:5]))
        return 0

    message = load_message(args)
    parse_mode = None if args.parse_mode == "None" else args.parse_mode
    inline_keyboard = load_inline_keyboard(args.inline_keyboard)
    photos = load_photos(args.photo)

    extra_params = {}
    if args.extra_api_params:
        try:
            extra_params = json.loads(args.extra_api_params)
        except json.JSONDecodeError as exc:
            print(f"Неверный JSON в extra-api-params: {exc}", file=sys.stderr)
            return 1

    config = TelegramBroadcastConfig(
        token=args.token,
        message=message,
        parse_mode=parse_mode,
        disable_web_page_preview=args.disable_preview,
        inline_keyboard=inline_keyboard,
        photos=photos,
        attach_message_to_first_photo=args.attach_message_to_first_photo,
        extra_api_params=extra_params,
    )

    sender = TelegramSender(token=args.token)
    summary = sender.broadcast(chat_ids, config)

    if summary.success:
        print(f"Готово! Сообщения доставлены {summary.delivered} из {summary.total}.")
        return 0

    print(
        f"Рассылка завершена с ошибками. Успешно: {summary.delivered}/{summary.total}",
        file=sys.stderr,
    )
    for report in summary.failed:
        print(f" - {report.chat_id}: {report.error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

