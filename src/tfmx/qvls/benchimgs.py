"""Image and prompt generation utilities for QVL benchmarking.

Loads real images from HuggingFace datasets (or local data directory)
and pairs them with diverse prompts for benchmarking Qwen3-VL
chat completion services.

Supports:
- Real images from HuggingFace datasets (downloaded to data/ dir)
- Local image files from the data directory
- Base64-encoded images for OpenAI-compatible API
- Text-only prompt generation
- Diverse vision-language test cases
"""

import base64
import io
import os
import random
from pathlib import Path
from typing import Optional

from tclogger import logger, logstr


# Default data directory for benchmark images
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
BENCH_IMAGES_DIR = DATA_DIR / "bench_images"

# Vision-language prompt templates
VL_PROMPTS = [
    "Describe what you see in this image in detail.",
    "What objects can you identify in this image?",
    "Describe the colors and shapes present in this image.",
    "What is the dominant element in this image?",
    "How many distinct objects can you see in this image?",
    "Describe the spatial arrangement of elements in this image.",
    "What mood or feeling does this image convey?",
    "If this were a painting, what style would it be?",
    "Describe the scene in this image as if explaining to someone who cannot see.",
    "What patterns do you notice in this image?",
    "Compare the different elements in this image.",
    "What is unusual or interesting about this image?",
    "Describe the texture and visual quality of this image.",
    "Summarize the visual content of this image in one sentence.",
    "What story does this image tell?",
    "Identify any text visible in this image.",
    "What time of day does this image suggest?",
    "Describe the composition and framing of this image.",
    "What emotions does this image evoke?",
    "If you had to give this image a title, what would it be?",
]

# Chinese vision-language prompts
CN_VL_PROMPTS = [
    "请详细描述这张图片中你看到的内容。",
    "你能识别这张图片中的哪些物体？",
    "描述这张图片中的颜色和形状。",
    "这张图片中最突出的元素是什么？",
    "描述这张图片中各元素的空间布局。",
    "这张图片传达了什么样的情绪或感觉？",
    "用一句话总结这张图片的视觉内容。",
    "这张图片讲述了什么故事？",
    "描述这张图片的构图和取景。",
    "如果给这张图片起个标题，你会叫它什么？",
]

# Text-only prompts for benchmarking without images
TEXT_PROMPTS = [
    "Explain the concept of artificial intelligence in simple terms.",
    "What are the main differences between machine learning and deep learning?",
    "Describe the process of photosynthesis step by step.",
    "Write a short poem about technology and nature.",
    "What are the key principles of good software design?",
    "Explain how neural networks work.",
    "What are the advantages and disadvantages of renewable energy?",
    "Describe the water cycle in detail.",
    "What makes a good programming language?",
    "Explain the concept of recursion with an example.",
    "What are the ethical considerations of AI development?",
    "Describe the history of the internet in brief.",
    "What is the difference between TCP and UDP?",
    "Explain the concept of version control.",
    "What are the principles of clean code?",
    "Describe the process of compiling code.",
    "What is containerization and why is it useful?",
    "Explain the CAP theorem in distributed systems.",
    "What are design patterns in software engineering?",
    "Describe the concept of microservices architecture.",
]

# Chinese text-only prompts
CN_TEXT_PROMPTS = [
    "请简要介绍人工智能的发展历史。",
    "深度学习和传统机器学习有什么区别？",
    "请描述光合作用的过程。",
    "写一首关于科技与自然的短诗。",
    "什么是良好的软件设计原则？",
    "请解释神经网络的工作原理。",
    "可再生能源的优缺点是什么？",
    "请详细描述水循环过程。",
    "编程语言应该具备哪些特点？",
    "用一个例子解释递归的概念。",
    "AI开发中的伦理考虑有哪些？",
    "简要介绍互联网的发展历史。",
    "TCP和UDP有什么区别？",
    "请解释版本控制的概念。",
    "什么是整洁代码的原则？",
]

# HuggingFace dataset configs for downloading benchmark images
HF_IMAGE_DATASETS = [
    {
        "name": "visual7w",
        "repo": "lmms-lab/Visual7W",
        "split": "test",
        "image_key": "image",
        "max_samples": 500,
    },
]

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def download_benchmark_images(
    output_dir: Path = BENCH_IMAGES_DIR,
    max_images: int = 500,
    dataset_name: str = "visual7w",
) -> int:
    """Download benchmark images from HuggingFace datasets.

    Args:
        output_dir: Directory to save images
        max_images: Maximum number of images to download
        dataset_name: Which dataset config to use

    Returns:
        Number of images downloaded
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warn(
            "× 'datasets' package not installed. " "Install with: pip install datasets"
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find matching dataset config
    ds_config = None
    for cfg in HF_IMAGE_DATASETS:
        if cfg["name"] == dataset_name:
            ds_config = cfg
            break

    if ds_config is None:
        logger.warn(f"× Unknown dataset: {dataset_name}")
        return 0

    logger.mesg(f"[benchimgs] Downloading images from {ds_config['repo']}...")
    logger.mesg(f"[benchimgs] Output dir: {output_dir}")

    try:
        ds = load_dataset(
            ds_config["repo"],
            split=ds_config["split"],
            streaming=True,
        )
    except Exception as e:
        logger.warn(f"× Failed to load dataset: {e}")
        return 0

    count = 0
    for item in ds:
        if count >= max_images:
            break

        try:
            image = item.get(ds_config["image_key"])
            if image is None:
                continue

            out_path = output_dir / f"{dataset_name}_{count:04d}.jpg"
            if hasattr(image, "save"):
                image.save(out_path, format="JPEG", quality=85)
            elif isinstance(image, bytes):
                out_path.write_bytes(image)
            else:
                continue

            count += 1
            if count % 100 == 0:
                logger.mesg(f"[benchimgs] Downloaded {count}/{max_images} images")

        except Exception as e:
            logger.warn(f"× Error saving image {count}: {e}")
            continue

    logger.okay(f"[benchimgs] Downloaded {count} images to {output_dir}")
    return count


def load_local_images(
    image_dir: Path = BENCH_IMAGES_DIR,
    max_images: int | None = None,
) -> list[Path]:
    """Load image file paths from a local directory.

    Args:
        image_dir: Directory containing images
        max_images: Maximum number of images to load (None = all)

    Returns:
        List of image file paths
    """
    if not image_dir.exists():
        return []

    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if max_images is not None:
        images = images[:max_images]

    return images


def _image_to_data_url(image_path: Path) -> str:
    """Convert a local image file to base64 data URL."""
    suffix = image_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _generate_synthetic_image_bytes(
    size: int = 256,
    rng: random.Random | None = None,
    fmt: str = "PNG",
) -> bytes:
    """Generate a synthetic image with random geometric shapes (fallback)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return _minimal_png(size)

    if rng is None:
        rng = random.Random()

    bg_colors = [
        (240, 240, 240),
        (255, 248, 220),
        (230, 230, 250),
        (245, 245, 220),
        (255, 250, 240),
        (240, 255, 240),
    ]
    shape_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
    ]

    bg = rng.choice(bg_colors)
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)

    for _ in range(rng.randint(2, 8)):
        color = rng.choice(shape_colors)
        x1 = rng.randint(0, size - 20)
        y1 = rng.randint(0, size - 20)
        x2 = rng.randint(x1 + 10, min(x1 + size // 2, size))
        y2 = rng.randint(y1 + 10, min(y1 + size // 2, size))

        shape = rng.choice(["rect", "ellipse"])
        if shape == "rect":
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=color)

    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _minimal_png(size: int = 8) -> bytes:
    """Generate a minimal solid-color PNG without PIL."""
    import struct
    import zlib

    width, height = size, size
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00" + b"\xff\x80\x80" * width

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + chunk
            + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = make_chunk(b"IHDR", ihdr_data)
    compressed = zlib.compress(raw_data)
    idat = make_chunk(b"IDAT", compressed)
    iend = make_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def _image_bytes_to_data_url(img_bytes: bytes, mime_type: str = "image/png") -> str:
    """Convert image bytes to data URL."""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


class QVLBenchImageGenerator:
    """Generate benchmark requests for QVL services.

    Uses real images from local data directory (downloaded from
    HuggingFace datasets) paired with diverse prompts.
    Falls back to synthetic images if no real images available.
    """

    def __init__(
        self,
        seed: int = 42,
        image_dir: Path | None = None,
    ):
        self.rng = random.Random(seed)
        self.image_dir = image_dir or BENCH_IMAGES_DIR
        self._local_images: list[Path] | None = None

    def _get_local_images(self) -> list[Path]:
        """Load and cache local image paths."""
        if self._local_images is None:
            self._local_images = load_local_images(self.image_dir)
        return self._local_images

    def _get_image_data_url(self, index: int, img_size: int = 256) -> str:
        """Get image data URL - from local files or synthetic fallback."""
        local_images = self._get_local_images()
        if local_images:
            img_path = local_images[index % len(local_images)]
            return _image_to_data_url(img_path)
        else:
            img_bytes = _generate_synthetic_image_bytes(size=img_size, rng=self.rng)
            return _image_bytes_to_data_url(img_bytes)

    def generate(
        self,
        count: int = 100,
        img_size: int = 256,
        img_format: str = "PNG",
    ) -> list[dict]:
        """Generate benchmark requests with images.

        Uses local real images if available, falls back to synthetic.

        Args:
            count: Number of requests to generate
            img_size: Image size (for synthetic fallback)
            img_format: Image format (for synthetic fallback)

        Returns:
            List of request dicts with 'messages' key
        """
        all_prompts = VL_PROMPTS + CN_VL_PROMPTS
        requests = []

        for i in range(count):
            prompt = self.rng.choice(all_prompts)
            data_url = self._get_image_data_url(i, img_size)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]
            requests.append({"messages": messages})

        return requests

    def generate_text_only(self, count: int = 100) -> list[dict]:
        """Generate text-only benchmark requests (no images).

        Args:
            count: Number of requests

        Returns:
            List of request dicts with 'messages' key
        """
        all_prompts = TEXT_PROMPTS + CN_TEXT_PROMPTS
        requests = []

        for i in range(count):
            prompt = self.rng.choice(all_prompts)
            messages = [{"role": "user", "content": prompt}]
            requests.append({"messages": messages})

        return requests

    def generate_mixed(
        self,
        count: int = 100,
        image_ratio: float = 0.7,
        img_size: int = 256,
    ) -> list[dict]:
        """Generate a mix of image and text-only requests.

        Args:
            count: Total number of requests
            image_ratio: Fraction of requests with images
            img_size: Image size for synthetic fallback

        Returns:
            List of request dicts with 'messages' key
        """
        n_image = int(count * image_ratio)
        n_text = count - n_image

        image_requests = self.generate(count=n_image, img_size=img_size)
        text_requests = self.generate_text_only(count=n_text)

        combined = image_requests + text_requests
        self.rng.shuffle(combined)
        return combined

    @property
    def local_image_count(self) -> int:
        """Number of local images available."""
        return len(self._get_local_images())

    @property
    def has_local_images(self) -> bool:
        """Whether local images are available."""
        return self.local_image_count > 0


def main():
    """CLI for downloading and testing benchmark images."""
    import argparse

    parser = argparse.ArgumentParser(
        description="QVL Benchmark Image Manager",
    )
    subparsers = parser.add_subparsers(dest="action")

    dl_parser = subparsers.add_parser("download", help="Download benchmark images")
    dl_parser.add_argument(
        "-n",
        "--max-images",
        type=int,
        default=500,
        help="Max images to download (default: 500)",
    )
    dl_parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="visual7w",
        help="Dataset name (default: visual7w)",
    )
    dl_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(BENCH_IMAGES_DIR),
        help=f"Output directory (default: {BENCH_IMAGES_DIR})",
    )

    subparsers.add_parser("info", help="Show image info")

    test_parser = subparsers.add_parser("test", help="Test image generation")
    test_parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=5,
        help="Number of samples to generate",
    )

    args = parser.parse_args()

    if args.action == "download":
        count = download_benchmark_images(
            output_dir=Path(args.output_dir),
            max_images=args.max_images,
            dataset_name=args.dataset,
        )
        logger.okay(f"Downloaded {count} images")

    elif args.action == "info":
        images = load_local_images()
        logger.note(f"> Benchmark images directory: {BENCH_IMAGES_DIR}")
        logger.mesg(f"  Images found: {len(images)}")
        if images:
            sizes = [p.stat().st_size for p in images[:100]]
            avg_size = sum(sizes) / len(sizes) / 1024
            logger.mesg(f"  Avg size: {avg_size:.1f} KB")
            logger.mesg(f"  First: {images[0].name}")
            logger.mesg(f"  Last: {images[-1].name}")

    elif args.action == "test":
        generator = QVLBenchImageGenerator(seed=42)
        logger.note(f"> Local images: {generator.local_image_count}")

        samples = generator.generate(count=args.count, img_size=64)
        logger.note(f"\n> Generated {len(samples)} image+prompt samples")
        for i, s in enumerate(samples):
            msgs = s["messages"]
            user_msg = msgs[0]
            content = user_msg["content"]
            text = next((p["text"] for p in content if p.get("type") == "text"), "?")
            source = "local" if generator.has_local_images else "synthetic"
            logger.mesg(f"  [{i + 1}] ({source}) {text[:50]}...")

        text_samples = generator.generate_text_only(count=args.count)
        logger.note(f"\n> Generated {len(text_samples)} text-only samples")
        for i, s in enumerate(text_samples):
            content = s["messages"][0]["content"]
            logger.mesg(f"  [{i + 1}] {content[:60]}...")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
