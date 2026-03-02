"""Image and prompt generation utilities for QVL benchmarking.

Generates synthetic image + prompt pairs for benchmarking
Qwen3-VL chat completion services.

Supports:
- Synthetic colored images with geometric shapes (via PIL)
- Base64-encoded images for OpenAI-compatible API
- Text-only prompt generation (no images)
- Diverse vision-language test cases
"""

import base64
import io
import random
from typing import Optional

from tclogger import logger, logstr


# Vision-language prompt templates
VL_PROMPTS = [
    "Describe what you see in this image in detail.",
    "What objects can you identify in this image?",
    "Describe the colors and shapes present in this image.",
    "What is the dominant color in this image?",
    "How many distinct shapes can you see in this image?",
    "Describe the spatial arrangement of elements in this image.",
    "What mood or feeling does this image convey?",
    "If this were a painting, what style would it be?",
    "Count the number of geometric shapes in this image.",
    "Describe this image as if you were explaining it to someone who cannot see.",
    "What patterns do you notice in this image?",
    "Compare the different elements in this image.",
    "What is unusual about this image?",
    "Describe the texture and visual quality of this image.",
    "Summarize the visual content of this image in one sentence.",
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

# Colors for synthetic image generation
COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 255, 255),
    (128, 128, 128),
    (64, 64, 64),
]

BG_COLORS = [
    (240, 240, 240),
    (255, 248, 220),
    (230, 230, 250),
    (245, 245, 220),
    (255, 250, 240),
    (240, 255, 240),
    (255, 240, 245),
    (240, 248, 255),
    (255, 255, 240),
]


def _generate_synthetic_image_bytes(
    size: int = 256,
    rng: random.Random | None = None,
    fmt: str = "PNG",
) -> bytes:
    """Generate a synthetic image with random geometric shapes.

    Creates a simple image using PIL with colored rectangles/ellipses
    on a pastel background.

    Args:
        size: Image width and height in pixels
        rng: Random number generator instance
        fmt: Image format (PNG, JPEG)

    Returns:
        Image bytes in specified format
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # Fallback: generate a minimal 1x1 PNG
        return _minimal_png(size)

    if rng is None:
        rng = random.Random()

    bg = rng.choice(BG_COLORS)
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)

    n_shapes = rng.randint(2, 8)
    for _ in range(n_shapes):
        color = rng.choice(COLORS)
        shape_type = rng.choice(["rect", "ellipse", "line"])

        x1 = rng.randint(0, size - 20)
        y1 = rng.randint(0, size - 20)
        x2 = rng.randint(x1 + 10, min(x1 + size // 2, size))
        y2 = rng.randint(y1 + 10, min(y1 + size // 2, size))

        if shape_type == "rect":
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)
        elif shape_type == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=None)
        elif shape_type == "line":
            width = rng.randint(2, 6)
            draw.line([x1, y1, x2, y2], fill=color, width=width)

    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _minimal_png(size: int = 8) -> bytes:
    """Generate a minimal solid-color PNG without PIL.

    Fallback for environments without Pillow installed.
    """
    import struct
    import zlib

    width, height = size, size
    # Create raw image data (filter byte + RGB per row)
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00"  # filter byte
        raw_data += b"\xff\x80\x80" * width  # salmon-colored pixels

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + chunk
            + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        )

    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = make_chunk(b"IHDR", ihdr_data)

    # IDAT chunk
    compressed = zlib.compress(raw_data)
    idat = make_chunk(b"IDAT", compressed)

    # IEND chunk
    iend = make_chunk(b"IEND", b"")

    return sig + ihdr + idat + iend


def _image_bytes_to_data_url(img_bytes: bytes, mime_type: str = "image/png") -> str:
    """Convert image bytes to data URL for OpenAI-compatible API."""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


class QVLBenchImageGenerator:
    """Generate synthetic benchmark requests for QVL services.

    Creates diverse image + prompt pairs or text-only prompts
    for benchmarking Qwen3-VL chat completion throughput.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(
        self,
        count: int = 100,
        img_size: int = 256,
        img_format: str = "PNG",
    ) -> list[dict]:
        """Generate benchmark requests with synthetic images.

        Args:
            count: Number of requests to generate
            img_size: Image size in pixels
            img_format: Image format (PNG/JPEG)

        Returns:
            List of request dicts with 'messages' key
        """
        requests = []

        for i in range(count):
            prompt = self.rng.choice(VL_PROMPTS)
            img_bytes = _generate_synthetic_image_bytes(
                size=img_size, rng=self.rng, fmt=img_format
            )
            data_url = _image_bytes_to_data_url(img_bytes)

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
            image_ratio: Fraction of requests with images (0.0-1.0)
            img_size: Image size in pixels

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


def main():
    """Quick test of benchmark image generation."""
    generator = QVLBenchImageGenerator(seed=42)

    # Generate a few samples
    samples = generator.generate(count=5, img_size=64)
    logger.note(f"> Generated {len(samples)} image+prompt samples")
    for i, s in enumerate(samples):
        msgs = s["messages"]
        user_msg = msgs[0]
        content = user_msg["content"]
        text = next((p["text"] for p in content if p.get("type") == "text"), "?")
        img_url = next(
            (
                p["image_url"]["url"][:60]
                for p in content
                if p.get("type") == "image_url"
            ),
            "?",
        )
        logger.mesg(f"  [{i + 1}] {text[:50]}... img={img_url}...")

    # Text-only samples
    text_samples = generator.generate_text_only(count=5)
    logger.note(f"\n> Generated {len(text_samples)} text-only samples")
    for i, s in enumerate(text_samples):
        content = s["messages"][0]["content"]
        logger.mesg(f"  [{i + 1}] {content[:60]}...")


if __name__ == "__main__":
    main()
