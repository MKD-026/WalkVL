#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import textwrap
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

from run_internvl import (
    build_segmentation_inputs,
    colorize_segmentation_mask,
    load_image,
    load_model,
    load_question_from_json,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run InternVL on frame folders and render a summary video."
    )
    parser.add_argument(
        "--input-mode",
        choices=["rgb_seg", "rgb_only"],
        default="rgb_seg",
        help="Inference mode: RGB+SEG or RGB only.",
    )
    parser.add_argument("--rgb-dir", required=True, help="Directory with RGB frames.")
    parser.add_argument(
        "--seg-dir",
        default=None,
        help="Directory with segmentation masks (required for --input-mode rgb_seg).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for results, vis frames, and video.",
    )
    parser.add_argument(
        "--model",
        default="OpenGVLab/InternVL2-8B",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--question",
        default="Use the RGB image and segmentation mask to describe each main region.",
        help="Prompt text asked to the model.",
    )
    parser.add_argument(
        "--question-json",
        default=None,
        help="Optional JSON file containing prompt text.",
    )
    parser.add_argument(
        "--question-key",
        default="question",
        help="JSON key used with --question-json.",
    )
    parser.add_argument(
        "--seg-mode",
        choices=["separate", "overlay", "both"],
        default="separate",
        help="How to feed RGB + seg to InternVL.",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.45,
        help="Alpha for RGB+SEG overlay in [0,1].",
    )
    parser.add_argument("--max-num", type=int, default=12, help="Max tiles per image.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max new tokens per frame."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Enable sampling (default deterministic).",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument(
        "--no-flash-attn", action="store_true", help="Disable flash attention."
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Load model with bitsandbytes 8-bit."
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument(
        "--max-frames", type=int, default=0, help="0 means use all; otherwise cap frames."
    )
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS.")
    parser.add_argument(
        "--panel-size",
        type=int,
        default=512,
        help="Size of each visual panel in output video.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining frames if one frame fails.",
    )
    return parser.parse_args()


def list_paired_frames(rgb_dir: Path, seg_dir: Path):
    rgb_files = sorted(
        [p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )
    pairs = []
    for rgb in rgb_files:
        seg = seg_dir / rgb.name
        if seg.exists() and seg.is_file():
            pairs.append((rgb, seg))
    return pairs


def list_rgb_frames(rgb_dir: Path):
    return sorted([p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def build_question(base_question: str, image_roles):
    if len(image_roles) == 1:
        return base_question if "<image>" in base_question else f"<image>\n{base_question}"
    tags = "\n".join(
        [f"Image-{idx + 1} ({role}): <image>" for idx, role in enumerate(image_roles)]
    )
    return f"{tags}\n{base_question}"


def wrap_text(text: str, width: int = 90):
    compact = " ".join(text.split())
    return textwrap.fill(compact, width=width)


def make_vis_frame(
    rgb_path: Path,
    seg_path,
    response_text: str,
    out_path: Path,
    panel_size: int,
    mask_alpha: float,
    frame_name: str,
):
    rgb = Image.open(rgb_path).convert("RGB")
    has_seg = seg_path is not None and Path(seg_path).exists()
    if has_seg:
        seg_raw = Image.open(seg_path)
        if seg_raw.size != rgb.size:
            seg_raw = seg_raw.resize(rgb.size, resample=Image.Resampling.NEAREST)
        seg_col = colorize_segmentation_mask(seg_raw)
        overlay = Image.blend(rgb, seg_col, alpha=mask_alpha)
    else:
        seg_col = Image.new("RGB", rgb.size, color=(40, 40, 40))
        overlay = Image.new("RGB", rgb.size, color=(40, 40, 40))

    rgb = rgb.resize((panel_size, panel_size), resample=Image.Resampling.BILINEAR)
    seg_col = seg_col.resize((panel_size, panel_size), resample=Image.Resampling.NEAREST)
    overlay = overlay.resize((panel_size, panel_size), resample=Image.Resampling.BILINEAR)

    header_h = 44
    text_h = 220
    canvas_w = panel_size * 3
    canvas_h = header_h + panel_size + text_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 22)
        font_text = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    canvas.paste(rgb, (0, header_h))
    canvas.paste(seg_col, (panel_size, header_h))
    canvas.paste(overlay, (panel_size * 2, header_h))

    draw.text((16, 10), "RGB", fill=(240, 240, 240), font=font_title)
    if has_seg:
        draw.text(
            (panel_size + 16, 10),
            "SEG (pseudo-color)",
            fill=(240, 240, 240),
            font=font_title,
        )
        draw.text(
            (panel_size * 2 + 16, 10),
            "RGB + SEG overlay",
            fill=(240, 240, 240),
            font=font_title,
        )
    else:
        draw.text((panel_size + 16, 10), "SEG (N/A)", fill=(180, 180, 180), font=font_title)
        draw.text(
            (panel_size * 2 + 16, 10),
            "RGB + SEG (N/A)",
            fill=(180, 180, 180),
            font=font_title,
        )
        draw.text(
            (panel_size + 20, header_h + panel_size // 2),
            "No segmentation input",
            fill=(220, 220, 220),
            font=font_text,
        )
        draw.text(
            (panel_size * 2 + 20, header_h + panel_size // 2),
            "No overlay available",
            fill=(220, 220, 220),
            font=font_text,
        )

    text_top = header_h + panel_size + 10
    draw.rectangle(
        [(0, header_h + panel_size), (canvas_w, canvas_h)],
        fill=(26, 26, 26),
        outline=(60, 60, 60),
        width=1,
    )
    title = f"Frame {frame_name} | InternVL response"
    draw.text((16, text_top), title, fill=(255, 220, 120), font=font_title)
    wrapped = wrap_text(response_text, width=100)
    draw.text((16, text_top + 34), wrapped, fill=(235, 235, 235), font=font_text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def extract_json_from_text(text: str):
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def build_analysis(records):
    total = len(records)
    parse_ok = 0
    safe_direction = Counter()
    risk_levels = Counter()
    stop_now_true = 0
    uncertainty_vals = []
    latencies = []

    for rec in records:
        if "latency_s" in rec:
            latencies.append(rec["latency_s"])
        if rec.get("error"):
            continue
        payload = extract_json_from_text(rec.get("response", ""))
        if not isinstance(payload, dict):
            continue
        parse_ok += 1
        direction = payload.get("safe_direction")
        if isinstance(direction, str):
            safe_direction[direction] += 1
        stop_now = payload.get("stop_now")
        if stop_now is True:
            stop_now_true += 1
        uncertainty = payload.get("uncertainty")
        if isinstance(uncertainty, (int, float)):
            uncertainty_vals.append(float(uncertainty))
        pedestrians = payload.get("pedestrians", [])
        if isinstance(pedestrians, list):
            for ped in pedestrians:
                if isinstance(ped, dict):
                    risk = ped.get("risk")
                    if isinstance(risk, str):
                        risk_levels[risk] += 1

    analysis = {
        "total_frames": total,
        "json_parse_success_frames": parse_ok,
        "json_parse_success_rate": (parse_ok / total) if total else 0.0,
        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "safe_direction_counts": dict(safe_direction),
        "stop_now_true_frames": stop_now_true,
        "mean_uncertainty": (
            sum(uncertainty_vals) / len(uncertainty_vals) if uncertainty_vals else None
        ),
        "pedestrian_risk_counts": dict(risk_levels),
    }
    return analysis


def write_analysis_text(analysis, out_path: Path):
    lines = []
    lines.append(f"Total frames: {analysis['total_frames']}")
    lines.append(
        "Structured JSON frames: "
        f"{analysis['json_parse_success_frames']} ({analysis['json_parse_success_rate']:.1%})"
    )
    if analysis["avg_latency_s"] is not None:
        lines.append(f"Average latency/frame: {analysis['avg_latency_s']:.2f}s")
    lines.append(f"Stop-now frames: {analysis['stop_now_true_frames']}")
    if analysis["mean_uncertainty"] is not None:
        lines.append(f"Mean uncertainty: {analysis['mean_uncertainty']:.3f}")
    lines.append(f"Safe-direction counts: {analysis['safe_direction_counts']}")
    lines.append(f"Pedestrian risk counts: {analysis['pedestrian_risk_counts']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_video(ffmpeg_bin: str, vis_dir: Path, fps: int, out_mp4: Path):
    cmd = [
        ffmpeg_bin,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(vis_dir / "%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    rgb_dir = Path(args.rgb_dir)
    seg_dir = Path(args.seg_dir) if args.seg_dir else None
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "vis_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not rgb_dir.exists():
        raise FileNotFoundError("--rgb-dir must exist.")
    if args.input_mode == "rgb_seg":
        if seg_dir is None:
            raise ValueError("--seg-dir is required when --input-mode rgb_seg.")
        if not seg_dir.exists():
            raise FileNotFoundError("--seg-dir must exist for --input-mode rgb_seg.")
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be in [0,1].")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1.")

    question = args.question
    if args.question_json:
        question = load_question_from_json(Path(args.question_json), key=args.question_key)

    if args.input_mode == "rgb_seg":
        pairs = list_paired_frames(rgb_dir, seg_dir)
        if not pairs:
            raise RuntimeError("No matching rgb/seg file pairs found.")
    else:
        rgb_files = list_rgb_frames(rgb_dir)
        if not rgb_files:
            raise RuntimeError("No RGB files found.")
        pairs = [(rgb, None) for rgb in rgb_files]
    pairs = pairs[:: args.frame_step]
    if args.max_frames > 0:
        pairs = pairs[: args.max_frames]

    print(f"Paired frames to process: {len(pairs)}")
    print(f"Output directory: {output_dir}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Run this inside a GPU session.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype={dtype}, cuda_device={torch.cuda.get_device_name(0)}")

    model = load_model(
        model_id=args.model,
        dtype=dtype,
        use_flash_attn=not args.no_flash_attn,
        load_in_8bit=args.load_in_8bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False
    )

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.sample,
    }
    if args.sample:
        generation_config["temperature"] = args.temperature

    results_jsonl = output_dir / "results.jsonl"
    records = []
    with results_jsonl.open("w", encoding="utf-8") as jf:
        for idx, (rgb_path, seg_path) in enumerate(pairs):
            frame_name = rgb_path.name
            started = time.time()
            record = {
                "idx": idx,
                "frame_name": frame_name,
                "rgb_path": str(rgb_path),
                "seg_path": str(seg_path) if seg_path else None,
                "input_mode": args.input_mode,
            }
            try:
                if args.input_mode == "rgb_seg":
                    model_images, image_roles = build_segmentation_inputs(
                        rgb_path=rgb_path,
                        mask_path=seg_path,
                        seg_mode=args.seg_mode,
                        mask_alpha=args.mask_alpha,
                    )
                else:
                    model_images = [Image.open(rgb_path).convert("RGB")]
                    image_roles = ["RGB image"]

                question_full = build_question(question, image_roles)
                batches = []
                num_patches_list = []
                for image_input in model_images:
                    pixel_values = load_image(image_input, max_num=args.max_num)
                    num_patches_list.append(pixel_values.size(0))
                    batches.append(pixel_values.to(dtype).cuda())
                pixel_values = torch.cat(batches, dim=0)
                extra_kwargs = {}
                if len(model_images) > 1:
                    extra_kwargs["num_patches_list"] = num_patches_list

                output = model.chat(
                    tokenizer,
                    pixel_values,
                    question_full,
                    generation_config,
                    history=None,
                    return_history=False,
                    **extra_kwargs,
                )
                response = output[0] if isinstance(output, tuple) else output
                record["response"] = str(response)
            except Exception as exc:
                record["error"] = str(exc)
                if not args.continue_on_error:
                    raise
            finally:
                record["latency_s"] = round(time.time() - started, 3)

            response_for_vis = record.get("response", f"ERROR: {record.get('error', 'unknown')}")
            vis_path = vis_dir / f"{idx:06d}.png"
            make_vis_frame(
                rgb_path=rgb_path,
                seg_path=seg_path,
                response_text=response_for_vis,
                out_path=vis_path,
                panel_size=args.panel_size,
                mask_alpha=args.mask_alpha,
                frame_name=frame_name,
            )

            records.append(record)
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")
            jf.flush()
            print(
                f"[{idx + 1}/{len(pairs)}] {frame_name} | "
                f"latency={record['latency_s']}s | error={'error' in record}"
            )

    analysis = build_analysis(records)
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_analysis_text(analysis, output_dir / "analysis.txt")

    ffmpeg_bin = subprocess.run(
        ["which", "ffmpeg"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found in PATH.")
    video_path = output_dir / "summary.mp4"
    render_video(ffmpeg_bin, vis_dir, args.fps, video_path)

    print("\nDone.")
    print(f"- Results JSONL: {results_jsonl}")
    print(f"- Analysis TXT : {output_dir / 'analysis.txt'}")
    print(f"- Analysis JSON: {output_dir / 'analysis.json'}")
    print(f"- Video        : {video_path}")


if __name__ == "__main__":
    main()
