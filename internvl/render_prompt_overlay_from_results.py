#!/usr/bin/env python3
import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from batch_walkvlm_filtered import build_question, make_composite


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild prompt/response overlay frames from existing results.jsonl without rerunning inference."
    )
    parser.add_argument("--results", required=True, help="Path to results.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered PNG frames")
    parser.add_argument("--mode", choices=["rgb", "rgb_seg_depth"], default=None)
    parser.add_argument("--panel-width", type=int, default=512)
    parser.add_argument("--max-prompt-chars", type=int, default=650)
    parser.add_argument("--max-response-chars", type=int, default=900)
    parser.add_argument(
        "--composite-dir",
        default=None,
        help="Optional directory for RGB+Seg+Depth visualization composites.",
    )
    return parser.parse_args()


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compact(text):
    return " ".join(str(text or "").split())


def truncate(text, max_chars):
    text = compact(text)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def wrapped(text, width):
    return textwrap.fill(compact(text), width=width)


def draw_wrapped(draw, text, xy, font, fill, width_chars, line_spacing=4):
    x, y = xy
    for line in wrapped(text, width_chars).splitlines():
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + line_spacing
    return y


def image_for_record(record, mode, composite_dir):
    if mode == "rgb":
        return Path(record["rgb_path"])

    if record.get("vis_image") and Path(record["vis_image"]).exists():
        return Path(record["vis_image"])

    out_dir = Path(composite_dir) if composite_dir else Path(record["rgb_path"]).parent / "_prompt_overlay_composites"
    out_path = out_dir / f"{record.get('key') or Path(record['rgb_path']).stem}.png"
    if not out_path.exists():
        make_composite(
            Path(record["rgb_path"]),
            Path(record["seg_path"]),
            Path(record["depth_path"]),
            out_path,
        )
    return out_path


def render_frame(record, mode, image_path, out_path, panel_width, max_prompt_chars, max_response_chars):
    image = Image.open(image_path).convert("RGB").copy()
    aspect_h = max(1, round(panel_width * image.height / image.width))
    image = image.resize((panel_width, aspect_h), Image.Resampling.BILINEAR)

    prompt = record.get("prompt") or build_question(mode)
    response = record.get("response") or f"ERROR: {record.get('error', 'missing response')}"

    image_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    prompt_h = min(max(96, aspect_h // 4), 240)
    overlay_draw.rectangle([(0, 0), (panel_width, prompt_h)], fill=(0, 0, 0, 180))
    image = Image.alpha_composite(image_rgba, overlay).convert("RGB")

    text_h = 300
    canvas = Image.new("RGB", (panel_width, aspect_h + text_h), (22, 22, 22))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 22)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((14, 10), f"{mode} prompt", fill=(255, 220, 120), font=title_font)
    draw_wrapped(
        draw,
        truncate(prompt, max_prompt_chars),
        (14, 40),
        small_font,
        (245, 245, 245),
        width_chars=74,
        line_spacing=2,
    )

    draw.rectangle([(0, aspect_h), (panel_width, aspect_h + text_h)], fill=(28, 28, 28))
    draw.text(
        (14, aspect_h + 12),
        f"{mode} | {record.get('key') or record.get('frame_name')}",
        fill=(255, 220, 120),
        font=title_font,
    )
    draw_wrapped(
        draw,
        truncate(response, max_response_chars),
        (14, aspect_h + 48),
        body_font,
        (235, 235, 235),
        width_chars=78,
        line_spacing=4,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    args = parse_args()
    records = load_jsonl(args.results)
    if not records:
        raise RuntimeError(f"No records found in {args.results}")

    mode = args.mode or records[0].get("mode")
    if mode not in {"rgb", "rgb_seg_depth"}:
        raise ValueError("Could not infer mode; pass --mode rgb or --mode rgb_seg_depth.")

    output_dir = Path(args.output_dir)
    for idx, record in enumerate(records):
        image_path = image_for_record(record, mode, args.composite_dir)
        render_frame(
            record,
            mode,
            image_path,
            output_dir / f"{idx:06d}.png",
            args.panel_width,
            args.max_prompt_chars,
            args.max_response_chars,
        )
    print(f"Rendered {len(records)} overlay frames to {output_dir}")


if __name__ == "__main__":
    main()
