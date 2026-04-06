#!/usr/bin/env python3
"""Probe QWN multimodal chat with multiple images and text segments."""

import argparse
import base64
import struct
import zlib

from tfmx.qwns.client import QWNClient
from tfmx.qwns.client import build_multimodal_messages
from tfmx.qwns.client import format_stream_stats_line


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def make_solid_png_data_url(red: int, green: int, blue: int, size: int = 48) -> str:
    row = b"\x00" + bytes([red, green, blue]) * size
    raw = row * size
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw)),
            _png_chunk(b"IEND", b""),
        ]
    )
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe QWN multimodal messages")
    parser.add_argument(
        "-e",
        "--endpoint",
        default="http://localhost:27800",
        help="QWN machine endpoint",
    )
    parser.add_argument(
        "texts",
        nargs="*",
        default=[
            "请先看第一张图，然后描述你看到的主色调。",
            "再看第二张图，比较两张图的差异，并说明你一共收到了几张图片。",
        ],
        help="Prompt text segments",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Keep model thinking output enabled",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = [
        make_solid_png_data_url(220, 40, 40),
        make_solid_png_data_url(40, 80, 220),
    ]
    messages = build_multimodal_messages(texts=args.texts, images=images)

    with QWNClient(endpoint=args.endpoint) as client:
        result = client.stream_chat(
            messages=messages,
            enable_thinking=args.thinking,
            on_text=lambda chunk: print(chunk, end="", flush=True),
        )
    if result.text and not result.text.endswith("\n"):
        print()
    print(format_stream_stats_line(result))


if __name__ == "__main__":
    main()
