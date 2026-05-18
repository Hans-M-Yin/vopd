#!/usr/bin/env python3
"""Preview a local Hugging Face dataset snapshot with optional image export."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import mimetypes
from pathlib import Path
from typing import Any

from datasets import load_dataset


SUPPORTED_SUFFIXES = (".parquet", ".json", ".jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a local Hugging Face dataset directory and export a small HTML gallery."
    )
    parser.add_argument("dataset_path", help="Local dataset file or directory")
    parser.add_argument("-n", "--num-examples", type=int, default=3, help="Number of examples to preview")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="tmp/dataset_preview",
        help="Directory for exported images and HTML preview",
    )
    parser.add_argument(
        "--truncate",
        type=int,
        default=0,
        help="Truncate preview text to this many characters. Use 0 to disable truncation.",
    )
    return parser.parse_args()


def discover_data_files(dataset_path: Path) -> tuple[str, list[str]]:
    if dataset_path.is_file():
        suffix = dataset_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {dataset_path}")
        loader = "parquet" if suffix == ".parquet" else "json"
        return loader, [str(dataset_path)]

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    files = sorted(
        str(path)
        for path in dataset_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and ".git" not in path.parts
    )
    if not files:
        raise FileNotFoundError(f"No parquet/json/jsonl files found under: {dataset_path}")

    has_parquet = any(path.endswith(".parquet") for path in files)
    loader = "parquet" if has_parquet else "json"
    filtered = [path for path in files if path.endswith(".parquet")] if has_parquet else files
    return loader, filtered


def load_local_dataset(dataset_path: Path):
    loader, files = discover_data_files(dataset_path)
    dataset_dict = load_dataset(loader, data_files=files)
    split_name = next(iter(dataset_dict.keys()))
    return dataset_dict[split_name], files, split_name


def short_text(value: Any, limit: int = 0) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=json_fallback)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def json_fallback(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<bytes: {len(value)} bytes>"
    if isinstance(value, bytearray):
        return f"<bytearray: {len(value)} bytes>"
    return repr(value)


def write_image_bytes(image_bytes: bytes, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_bytes)


def infer_image_suffix(image_bytes: bytes, fallback_suffix: str = ".png") -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if image_bytes.startswith(b"BM"):
        return ".bmp"
    if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
        return ".webp"
    return fallback_suffix


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "image/png"


def maybe_decode_base64_string(value: str) -> bytes | None:
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        return None


def extract_image_bytes(image_item: Any) -> tuple[bytes | None, str]:
    suffix = ".png"

    if isinstance(image_item, str):
        return maybe_decode_base64_string(image_item), suffix

    if isinstance(image_item, bytes):
        return image_item, suffix

    if isinstance(image_item, bytearray):
        return bytes(image_item), suffix

    if isinstance(image_item, memoryview):
        return image_item.tobytes(), suffix

    if hasattr(image_item, "save"):
        buffer = io.BytesIO()
        image_format = getattr(image_item, "format", None) or "PNG"
        image_item.save(buffer, format=image_format)
        suffix = f".{image_format.lower()}"
        return buffer.getvalue(), suffix

    if isinstance(image_item, dict):
        image_bytes = image_item.get("bytes")
        if isinstance(image_bytes, str):
            return maybe_decode_base64_string(image_bytes), suffix
        if isinstance(image_bytes, bytes):
            return image_bytes, suffix
        if isinstance(image_bytes, bytearray):
            return bytes(image_bytes), suffix
        if isinstance(image_bytes, memoryview):
            return image_bytes.tobytes(), suffix

    return None, suffix


def maybe_export_images(example: dict[str, Any], example_idx: int, output_dir: Path) -> list[dict[str, str]]:
    image_items: list[dict[str, str]] = []
    images = example.get("images")
    if not images:
        return image_items

    for image_idx, image_item in enumerate(images):
        image_bytes, suffix = extract_image_bytes(image_item)

        if image_bytes is None and isinstance(image_item, dict) and image_item.get("path"):
            path = Path(image_item["path"])
            if path.exists():
                resolved_path = path.resolve()
                image_items.append(
                    {
                        "path": str(resolved_path),
                        "src": resolved_path.as_uri(),
                        "mime_type": guess_mime_type(resolved_path),
                    }
                )
                continue

        if isinstance(image_item, dict):
            original_path = image_item.get("path")
            if isinstance(original_path, str):
                suffix = Path(original_path).suffix or suffix

        if not image_bytes:
            continue

        suffix = infer_image_suffix(image_bytes, suffix)
        output_path = output_dir / "images" / f"example_{example_idx:03d}_{image_idx:02d}{suffix}"
        write_image_bytes(image_bytes, output_path)
        relative_path = output_path.relative_to(output_dir).as_posix()
        image_items.append(
            {
                "path": str(output_path.resolve()),
                "src": relative_path,
                "mime_type": guess_mime_type(output_path),
            }
        )

    return image_items


def sanitize_example_for_preview(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_example_for_preview(val) for key, val in value.items()}
    if isinstance(value, list):
        return [sanitize_example_for_preview(item) for item in value]
    if isinstance(value, bytes):
        return f"<bytes: {len(value)} bytes>"
    if isinstance(value, bytearray):
        return f"<bytearray: {len(value)} bytes>"
    return value


def render_message_blocks(messages: Any, truncate: int) -> str:
    if not isinstance(messages, list):
        return (
            '<pre style="white-space:pre-wrap;word-break:break-word;background:#f7f7f7;'
            f'padding:12px;border-radius:10px;">{html.escape(short_text(sanitize_example_for_preview(messages), truncate))}</pre>'
        )

    blocks: list[str] = []
    for idx, message in enumerate(messages):
        role = message.get("role", f"message_{idx}") if isinstance(message, dict) else f"message_{idx}"
        content = message.get("content") if isinstance(message, dict) else message
        blocks.append(
            f"""
      <div style="margin:14px 0;padding:14px;border-radius:12px;background:#f7f7f7;">
        <div style="font-weight:600;margin-bottom:8px;">{html.escape(str(role))}</div>
        <pre style="margin:0;white-space:pre-wrap;word-break:break-word;">{html.escape(short_text(sanitize_example_for_preview(content), truncate))}</pre>
      </div>
"""
        )
    return "\n".join(blocks)


def render_example_card(example_idx: int, example: dict[str, Any], image_items: list[dict[str, str]], truncate: int) -> str:
    image_html = ""
    if image_items:
        image_html = "\n".join(
            f'<img src="{html.escape(image_item["src"])}" '
            'style="max-width: 420px; max-height: 320px; border-radius: 12px; display: block; margin-bottom: 12px;" />'
            for image_item in image_items
        )

    raw_preview = short_text(sanitize_example_for_preview(example), truncate)
    raw_images = example.get("images")
    raw_image_count = len(raw_images) if isinstance(raw_images, list) else 0
    return f"""
    <section style="border:1px solid #ddd;border-radius:16px;padding:20px;margin:20px 0;background:#fff;">
      <h2 style="margin-top:0;">Example {example_idx}</h2>
      <div style="margin-bottom:12px;color:#555;">raw images: {raw_image_count} | exported images: {len(image_items)}</div>
      {image_html}
      <h3>messages</h3>
      {render_message_blocks(example.get("messages", example), truncate)}
      <h3>raw row</h3>
      <pre style="white-space:pre-wrap;word-break:break-word;background:#f7f7f7;padding:12px;border-radius:10px;">{html.escape(raw_preview)}</pre>
    </section>
    """


def write_html_preview(output_dir: Path, cards: list[str], summary: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "preview.html"
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Dataset Preview</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#f2f4f8; color:#111; margin:0; padding:32px;">
  <main style="max-width: 1000px; margin: 0 auto;">
    <h1>Dataset Preview</h1>
    <pre style="white-space:pre-wrap;word-break:break-word;background:#fff;padding:16px;border-radius:12px;border:1px solid #ddd;">{html.escape(summary)}</pre>
    {''.join(cards)}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    truncate = max(0, args.truncate)

    dataset, files, split_name = load_local_dataset(dataset_path)
    num_examples = min(args.num_examples, len(dataset))

    summary = {
        "dataset_path": str(dataset_path),
        "split": split_name,
        "num_rows": len(dataset),
        "columns": dataset.column_names,
        "data_files": files,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    cards: list[str] = []
    for idx in range(num_examples):
        example = dataset[idx]
        image_items = maybe_export_images(example, idx, output_dir)
        print(f"\n===== Example {idx} =====")
        print(short_text(sanitize_example_for_preview(example), truncate))
        cards.append(render_example_card(idx, example, image_items, truncate))

    html_path = write_html_preview(output_dir, cards, json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nPreview written to: {html_path}")


if __name__ == "__main__":
    main()
