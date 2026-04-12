#!/usr/bin/env python3
"""Probe QSR transcription and chat flows against a machine or backend endpoint."""

import argparse
import json

from tfmx.qsrs.client import QSRClient, build_audio_messages, format_stream_stats_line


DEFAULT_AUDIO = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/" "Qwen3-ASR-Repo/asr_zh.wav"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe QSR ASR endpoints")
    parser.add_argument(
        "-e",
        "--endpoint",
        default="http://127.0.0.1:27900",
        help="QSR machine or backend endpoint",
    )
    parser.add_argument(
        "-a",
        "--audio",
        default=DEFAULT_AUDIO,
        help="Audio file path, URL, or data URI",
    )
    parser.add_argument(
        "--prompt",
        default="请转写为简体中文，并补一句简短摘要。",
        help="Prompt used for chat mode",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Only run transcription probe",
    )
    parser.add_argument(
        "--response-format",
        choices=["json", "text", "verbose_json", "srt", "vtt"],
        default="json",
        help="Transcription response format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with QSRClient(endpoint=args.endpoint, verbose=True) as client:
        print("=== health ===")
        print(client.health())
        print("=== models ===")
        print(client.models())
        print("=== transcribe ===")
        transcribed = client.transcribe(
            audio=args.audio,
            response_format=args.response_format,
        )
        print(json.dumps(transcribed.to_dict(), ensure_ascii=False, indent=2))

        if args.skip_chat:
            return

        print("=== chat ===")
        messages = build_audio_messages(texts=[args.prompt], audios=[args.audio])
        result = client.stream_chat(
            messages=messages,
            on_text=lambda chunk: print(chunk, end="", flush=True),
        )
        if result.text and not result.text.endswith("\n"):
            print()
        print(format_stream_stats_line(result))


if __name__ == "__main__":
    main()
