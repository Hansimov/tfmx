#!/usr/bin/env python3
"""Download Bilibili audio assets for QSR long-audio testing."""

import argparse
import json
import sys

from pathlib import Path


DEFAULT_BVIDS = [
    "BV1Gp9KBgEhc",
    "BV1uPDTBhEHX",
    "BV1ZD9PBGEwJ",
    "BV12n9uBMEv1",
    "BV1XL93BJEPT",
]
DEFAULT_OUTPUT_ROOT = Path("runs/qsrs/audio_inputs/bilibili")
DEFAULT_MANIFEST_PATH = Path("runs/qsrs/results/qsr_bili_audio_manifest.json")
DEFAULT_BLUX_SRC = Path("/home/asimov/repos/blux/src")
AUDIO_FORMAT_CHOICES = ("wav", "mp3", "m4a", "flac", "opus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Bilibili audio files for QSR long-audio benchmarks"
    )
    parser.add_argument("bvids", nargs="*", default=DEFAULT_BVIDS)
    parser.add_argument(
        "--blux-src",
        type=Path,
        default=DEFAULT_BLUX_SRC,
        help="Path containing the blux package source tree",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where downloaded audio artifacts are stored",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="JSON file that records download outputs and QSR-friendly paths",
    )
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--audio-strategy",
        choices=["lowest", "highest"],
        default="lowest",
        help="Pick lower bitrate for lighter local upload, or higher for fidelity",
    )
    parser.add_argument(
        "--convert-format",
        choices=AUDIO_FORMAT_CHOICES,
        default=None,
        help="Optionally convert downloaded audio via blux.convert",
    )
    parser.add_argument(
        "--audio-bitrate",
        default=None,
        help="Optional ffmpeg bitrate for converted audio, for example 24k",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=None,
        help="Optional output sample rate for converted audio",
    )
    parser.add_argument(
        "--audio-channels",
        type=int,
        default=None,
        help="Optional output channel count for converted audio",
    )
    parser.add_argument(
        "--drop-original-audio",
        action="store_true",
        help="Delete the raw downloaded audio after successful conversion",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def ensure_blux_import(blux_src: Path) -> None:
    resolved = blux_src.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"blux source path not found: {resolved}")
    sys.path.insert(0, str(resolved))


def main() -> None:
    args = parse_args()
    ensure_blux_import(args.blux_src)

    from blux.download.config import DownloadSettings
    from blux.download.downloader import BiliDownloader

    settings = DownloadSettings(
        output_root=args.output_root,
        audio_strategy=args.audio_strategy,
        audio_convert_format=args.convert_format,
        audio_convert_bitrate=args.audio_bitrate,
        audio_convert_sample_rate=args.audio_sample_rate,
        audio_convert_channels=args.audio_channels,
        keep_original_audio=not args.drop_original_audio,
        workers=args.workers,
        resume=not args.no_resume,
        page_index=args.page,
    )
    downloader = BiliDownloader(settings=settings)
    results = downloader.download_many(
        args.bvids,
        {"audio"},
        workers=args.workers,
        page_index=args.page,
        output_root=args.output_root,
        resume=not args.no_resume,
    )

    payload: list[dict] = []
    for result in results:
        audio_entries = []
        raw_audio_path = None
        converted_audio_path = None
        for artifact in result.artifacts:
            if artifact.kind == "audio":
                raw_audio_path = str(artifact.path)
            elif artifact.kind == "audio_converted":
                converted_audio_path = str(artifact.path)
            else:
                continue
            audio_entries.append(
                {
                    "kind": artifact.kind,
                    "path": str(artifact.path),
                    "bytes_written": artifact.bytes_written,
                    "source_url": artifact.source_url,
                }
            )
        qsr_audio_path = converted_audio_path or raw_audio_path
        payload.append(
            {
                "bvid": result.bvid,
                "title": result.title,
                "page_index": result.page_index,
                "output_dir": str(result.output_dir),
                "manifest_path": (
                    str(result.manifest_path) if result.manifest_path else None
                ),
                "raw_audio_path": raw_audio_path,
                "converted_audio_path": converted_audio_path,
                "qsr_input_path": qsr_audio_path,
                "audio": audio_entries,
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
