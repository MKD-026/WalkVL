#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import textwrap
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

from run_internvl import load_image, load_model


SEMANTIC_CLASSES = [
    (0, "background", (0, 0, 0), "#000000"),
    (1, "curb", (255, 214, 10), "#FFD60A"),
    (2, "sidewalk", (52, 199, 89), "#34C759"),
    (3, "crosswalk", (0, 122, 255), "#007AFF"),
    (4, "other_walkable", (100, 210, 255), "#64D2FF"),
    (5, "tree", (48, 176, 89), "#30B059"),
    (6, "road", (99, 99, 102), "#636366"),
    (7, "pedestrian", (255, 45, 85), "#FF2D55"),
    (8, "vehicle", (255, 149, 0), "#FF9500"),
    (9, "building", (175, 82, 222), "#AF52DE"),
    (10, "stairs", (191, 90, 242), "#BF5AF2"),
    (11, "obstacle", (255, 59, 48), "#FF3B30"),
]


RGB_PROMPT = """You are a pedestrian navigation assistant in the style of WalkGPT/WalkVLM.
You receive one front-facing RGB camera image from an autonomous navigation scene.
Analyze pedestrian safety, walkable space, obstacles, and the best immediate motion.

Return strict JSON only with these keys:
- scene_summary: one concise sentence
- pedestrians: list of objects with bbox [x1,y1,x2,y2] if visible, risk "low"|"med"|"high", crossing_intent 0-1
- walkable_regions: list of visible safe regions such as sidewalk_left, sidewalk_right, crosswalk, road_ahead, other_walkable
- hazards: list of visible hazards such as pedestrian, vehicle, curb, stairs, obstacle, tree, building_edge
- recommended_action: "move_forward"|"turn_left"|"turn_right"|"slow_down"|"stop"
- safe_direction: "left"|"right"|"forward"|"stop"
- stop_now: true|false
- uncertainty: float 0-1
- short_reason: one sentence grounded only in the RGB image"""


RGB_SEG_DEPTH_PROMPT = """You are a pedestrian navigation assistant in the style of WalkGPT/WalkVLM.
You receive three aligned images from the same front-facing scene:
Image-1: RGB scene.
Image-2: semantic segmentation with this legend:
0 background #000000, 1 curb #FFD60A, 2 sidewalk #34C759, 3 crosswalk #007AFF,
4 other_walkable #64D2FF, 5 tree #30B059, 6 road #636366, 7 pedestrian #FF2D55,
8 vehicle #FF9500, 9 building #AF52DE, 10 stairs #BF5AF2, 11 obstacle #FF3B30.
Image-3: grayscale depth map from Depth Anything V2, where relative brightness indicates relative depth.

Reason jointly across RGB, semantic classes, and depth. Use segmentation to identify walkable regions and hazards, and use depth to judge proximity.

Return strict JSON only with these keys:
- scene_summary: one concise sentence
- pedestrians: list of objects with risk "low"|"med"|"high", crossing_intent 0-1, distance "near"|"mid"|"far"
- walkable_regions: list of visible safe regions using class names from the legend
- hazards: list of visible hazards using class names from the legend
- depth_cues: one short sentence about near/mid/far safety-relevant regions
- recommended_action: "move_forward"|"turn_left"|"turn_right"|"slow_down"|"stop"
- safe_direction: "left"|"right"|"forward"|"stop"
- stop_now: true|false
- uncertainty: float 0-1
- short_reason: one sentence grounded in RGB, segmentation, and depth evidence"""


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run InternVL on filtered RGB and RGB+SEG+DEPTH frames.")
    parser.add_argument("--rgb-root", default="/projectnb/cs585/students/mkd/740/filtered")
    parser.add_argument("--seg-dir", default="/projectnb/cs585/students/mkd/740/semantic_results-2/segmented_images")
    parser.add_argument("--depth-dir", default="/projectnb/cs585/students/mkd/740/semantic_results-2/depth_maps")
    parser.add_argument("--mask-dir", default="/projectnb/cs585/students/mkd/740/semantic_results-2/annotation_masks")
    parser.add_argument("--output-root", default="/projectnb/cs585/students/mkd/740/WalkVL/experiments/internvl/outputs/filtered_walkvlm")
    parser.add_argument("--model", default="OpenGVLab/InternVL2-1B")
    parser.add_argument("--modes", nargs="+", choices=["rgb", "rgb_seg_depth"], default=["rgb", "rgb_seg_depth"])
    parser.add_argument("--routes", nargs="+", default=[], help="Optional route names/numbers to include, e.g. route1 route4 route5 or 1 4 5.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-num", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-flash-attn", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--panel-width", type=int, default=512)
    parser.add_argument("--max-prompt-chars", type=int, default=650)
    parser.add_argument("--no-scene-priors", action="store_true", help="Do not append computed mask/depth scene priors to RGB+Seg+Depth prompts.")
    parser.add_argument(
        "--rgb-prompt-json",
        default="/projectnb/cs585/students/mkd/740/WalkVL/internvl/prompts/walkvlm_conversational_rgb_prompt.json",
        help="JSON file with key 'question' for RGB mode.",
    )
    parser.add_argument(
        "--rgb-seg-depth-prompt-json",
        default="/projectnb/cs585/students/mkd/740/WalkVL/internvl/prompts/walkvlm_conversational_rgb_seg_depth_prompt.json",
        help="JSON file with key 'question' for RGB+Seg+Depth mode.",
    )
    return parser.parse_args()


def load_prompt_json(path: str, fallback: str):
    if not path:
        return fallback
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt JSON not found: {prompt_path}")
    with prompt_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Prompt JSON must contain a non-empty 'question': {prompt_path}")
    return question


def frame_key(path: Path):
    return path.stem


def find_rgb_frames(rgb_root: Path):
    frames = []
    for folder in sorted(rgb_root.glob("cropped_route*")):
        if folder.is_dir():
            frames.extend(sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS))
    return frames


def normalize_routes(routes):
    normalized = set()
    for route in routes or []:
        route = str(route).strip().lower()
        if not route:
            continue
        normalized.add(route if route.startswith("route") else f"route{route}")
    return normalized


def paired_records(args):
    rgb_root = Path(args.rgb_root)
    seg_dir = Path(args.seg_dir)
    depth_dir = Path(args.depth_dir)
    selected_routes = normalize_routes(args.routes)
    records = []
    for rgb_path in find_rgb_frames(rgb_root):
        route_name = rgb_path.parent.name.replace("cropped_", "", 1)
        if selected_routes and route_name not in selected_routes:
            continue
        key = frame_key(rgb_path)
        seg_path = seg_dir / f"segmented_{key}.jpg.png"
        depth_path = depth_dir / f"{key}.png"
        if seg_path.exists() and depth_path.exists():
            records.append({"key": key, "route": route_name, "rgb": rgb_path, "seg": seg_path, "depth": depth_path})
    records = records[:: args.frame_step]
    if args.max_frames > 0:
        records = records[: args.max_frames]
    return records


def make_composite(rgb_path: Path, seg_path: Path, depth_path: Path, out_path: Path):
    rgb = Image.open(rgb_path).convert("RGB")
    seg = Image.open(seg_path).convert("RGB").resize(rgb.size, Image.Resampling.NEAREST)
    depth = Image.open(depth_path).convert("RGB").resize(rgb.size, Image.Resampling.BILINEAR)
    composite = Image.new("RGB", (rgb.width, rgb.height * 3))
    composite.paste(rgb, (0, 0))
    composite.paste(seg, (0, rgb.height))
    composite.paste(depth, (0, rgb.height * 2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path)
    return out_path


CLASS_NAMES = {class_id: name for class_id, name, _, _ in SEMANTIC_CLASSES}
WALKABLE_CLASSES = {1, 2, 3, 4, 6}
SAFETY_CLASSES = {1, 7, 8, 10, 11}
CONTEXT_CLASSES = {5, 9}


def position_label(xs, width):
    center = float(xs.mean()) / max(1, width)
    if center < 0.33:
        return "left"
    if center > 0.67:
        return "right"
    return "center"


def vertical_label(ys, height):
    center = float(ys.mean()) / max(1, height)
    if center < 0.40:
        return "upper/far field"
    if center > 0.68:
        return "lower/near walking field"
    return "middle field"


def depth_label(values, near_cut, far_cut):
    median_value = float(np.median(values))
    if median_value >= near_cut:
        return "near/bright"
    if median_value <= far_cut:
        return "far/dark"
    return "mid"


def format_items(items, max_items=8):
    if not items:
        return "none"
    return "; ".join(items[:max_items])


def build_scene_priors(record, mask_dir: Path):
    mask_path = mask_dir / f"mask_{record['key']}.npy"
    if not mask_path.exists():
        return ""
    try:
        mask = np.load(mask_path)
        depth = np.asarray(Image.open(record["depth"]).convert("L").resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR))
    except Exception as exc:
        return f"Computed segmentation/depth priors unavailable: {exc}"

    height, width = mask.shape
    total_pixels = max(1, height * width)
    depth_present = depth[mask != 0] if np.any(mask != 0) else depth.reshape(-1)
    near_cut = float(np.percentile(depth_present, 70))
    far_cut = float(np.percentile(depth_present, 30))

    present = []
    walkable = []
    safety_relevant = []
    context = []
    ignored_far = []

    for class_id, class_name, _, _ in SEMANTIC_CLASSES:
        if class_id == 0:
            continue
        class_mask = mask == class_id
        pixel_count = int(class_mask.sum())
        if pixel_count == 0:
            continue
        area_pct = 100.0 * pixel_count / total_pixels
        if area_pct < 0.03:
            continue
        ys, xs = np.nonzero(class_mask)
        pos = position_label(xs, width)
        vertical = vertical_label(ys, height)
        depth_bucket = depth_label(depth[class_mask], near_cut, far_cut)
        item = f"{class_name}: {pos}, {vertical}, {depth_bucket}, area {area_pct:.1f}%"
        present.append(f"{class_name} ({area_pct:.1f}%)")
        if class_id in WALKABLE_CLASSES:
            walkable.append(item)
        if class_id in SAFETY_CLASSES:
            if depth_bucket == "far/dark" and vertical == "upper/far field":
                ignored_far.append(item)
            else:
                safety_relevant.append(item)
        elif class_id in CONTEXT_CLASSES:
            context.append(item)

    if not present:
        return ""

    return "\n".join(
        [
            "Computed segmentation/depth priors (hints from masks; verify against the images):",
            f"- Present semantic classes: {format_items(present, 12)}.",
            f"- Walkable/path regions: {format_items(walkable, 6)}.",
            f"- Nearby or path-relevant candidates: {format_items(safety_relevant, 6)}.",
            f"- Far/background context to avoid over-warning about: {format_items(ignored_far + context, 6)}.",
            "- Depth note: near/bright, mid, and far/dark are relative within this frame, not metric feet.",
        ]
    )


def build_question(mode: str, prompts=None, scene_priors: str = ""):
    prompts = prompts or {}
    if mode == "rgb":
        return f"<image>\n{prompts.get('rgb', RGB_PROMPT)}"
    question = (
        "Image-1 (RGB): <image>\n"
        "Image-2 (semantic segmentation): <image>\n"
        "Image-3 (Depth Anything V2 depth): <image>\n"
        f"{prompts.get('rgb_seg_depth', RGB_SEG_DEPTH_PROMPT)}"
    )
    if scene_priors:
        question += f"\n\n{scene_priors}"
    return question


def extract_json(text: str):
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
        try:
            return json.loads(cleaned[start : end + 1])
        except Exception:
            return None
    return None


def summarize(records):
    parsed = 0
    stop_count = 0
    actions = Counter()
    directions = Counter()
    uncertainties = []
    for rec in records:
        payload = extract_json(rec.get("response", ""))
        if not isinstance(payload, dict):
            continue
        parsed += 1
        if payload.get("stop_now") is True:
            stop_count += 1
        if isinstance(payload.get("recommended_action"), str):
            actions[payload["recommended_action"]] += 1
        if isinstance(payload.get("safe_direction"), str):
            directions[payload["safe_direction"]] += 1
        if isinstance(payload.get("uncertainty"), (int, float)):
            uncertainties.append(float(payload["uncertainty"]))
    total = len(records)
    return {
        "total_frames": total,
        "json_parse_success_frames": parsed,
        "json_parse_success_rate": parsed / total if total else 0.0,
        "stop_now_true_frames": stop_count,
        "recommended_action_counts": dict(actions),
        "safe_direction_counts": dict(directions),
        "mean_uncertainty": sum(uncertainties) / len(uncertainties) if uncertainties else None,
    }


def wrapped(text, width=90):
    return textwrap.fill(" ".join(text.split()), width=width)


def truncate_text(text: str, max_chars: int):
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def draw_text_block(draw, text, xy, font, fill, width_chars, line_spacing=4):
    x, y = xy
    for line in wrapped(text, width_chars).splitlines():
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + line_spacing
    return y


def make_vis_frame(
    record,
    image_path: Path,
    prompt: str,
    response: str,
    out_path: Path,
    panel_width: int,
    mode: str,
    max_prompt_chars: int,
):
    image = Image.open(image_path).convert("RGB").copy()
    aspect_h = max(1, round(panel_width * image.height / image.width))
    image = image.resize((panel_width, aspect_h), Image.Resampling.BILINEAR)
    image_rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", image_rgba.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    prompt_h = min(max(96, aspect_h // 4), 220)
    overlay_draw.rectangle([(0, 0), (panel_width, prompt_h)], fill=(0, 0, 0, 178))
    image_rgba = Image.alpha_composite(image_rgba, overlay)
    image = image_rgba.convert("RGB")

    text_h = 260
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
    prompt_title = f"{mode} prompt"
    draw.text((14, 10), prompt_title, fill=(255, 220, 120), font=title_font)
    draw_text_block(
        draw,
        truncate_text(prompt, max_prompt_chars),
        (14, 40),
        small_font,
        (245, 245, 245),
        width_chars=74,
        line_spacing=2,
    )
    draw.rectangle([(0, aspect_h), (panel_width, aspect_h + text_h)], fill=(28, 28, 28))
    draw.text((14, aspect_h + 12), f"{mode} | {record['key']}", fill=(255, 220, 120), font=title_font)
    draw.text((14, aspect_h + 46), wrapped(response, 80), fill=(235, 235, 235), font=body_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def render_video(vis_dir: Path, fps: int, out_path: Path):
    ffmpeg = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True, check=False).stdout.strip()
    if not ffmpeg:
        return
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i", str(vis_dir / "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
        check=True,
    )


def run_mode(args, mode: str, source_records, model, tokenizer, dtype, prompts):
    output_dir = Path(args.output_root) / mode
    vis_dir = output_dir / "vis_frames"
    composite_dir = output_dir / "composites"
    mask_dir = Path(args.mask_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_out = []
    generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": args.sample}
    if args.sample:
        generation_config["temperature"] = args.temperature

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for idx, rec in enumerate(source_records):
            started = time.time()
            image_paths = [rec["rgb"]]
            image_path = rec["rgb"]
            if mode == "rgb_seg_depth":
                image_path = make_composite(rec["rgb"], rec["seg"], rec["depth"], composite_dir / f"{rec['key']}.png")
                image_paths = [rec["rgb"], rec["seg"], rec["depth"]]
            scene_priors = ""
            if mode == "rgb_seg_depth" and not args.no_scene_priors:
                scene_priors = build_scene_priors(rec, mask_dir)
            prompt = build_question(mode, prompts, scene_priors)
            out_record = {
                "idx": idx,
                "frame_name": rec["rgb"].name,
                "key": rec["key"],
                "route": rec.get("route"),
                "mode": mode,
                "rgb_path": str(rec["rgb"]),
                "seg_path": str(rec["seg"]),
                "depth_path": str(rec["depth"]),
                "model_images": [str(path) for path in image_paths],
                "vis_image": str(image_path),
                "prompt": prompt,
                "scene_priors": scene_priors,
            }
            try:
                batches = []
                num_patches_list = []
                for model_image_path in image_paths:
                    image_pixels = load_image(model_image_path, max_num=args.max_num)
                    num_patches_list.append(image_pixels.size(0))
                    batches.append(image_pixels.to(dtype).cuda())
                pixel_values = torch.cat(batches, dim=0)
                extra_kwargs = {}
                if len(image_paths) > 1:
                    extra_kwargs["num_patches_list"] = num_patches_list
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    prompt,
                    generation_config,
                    history=None,
                    return_history=False,
                    **extra_kwargs,
                )
                out_record["response"] = str(response[0] if isinstance(response, tuple) else response)
            except Exception as exc:
                out_record["error"] = str(exc)
                if not args.continue_on_error:
                    raise
            finally:
                out_record["latency_s"] = round(time.time() - started, 3)
            make_vis_frame(
                rec,
                image_path,
                prompt,
                out_record.get("response", f"ERROR: {out_record.get('error', 'unknown')}"),
                vis_dir / f"{idx:06d}.png",
                args.panel_width,
                mode,
                args.max_prompt_chars,
            )
            handle.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            handle.flush()
            records_out.append(out_record)
            print(f"[{mode} {idx + 1}/{len(source_records)}] {rec['key']} latency={out_record['latency_s']}s error={'error' in out_record}", flush=True)

    analysis = summarize(records_out)
    (output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    render_video(vis_dir, args.fps, output_dir / "summary.mp4")


def main():
    args = parse_args()
    records = paired_records(args)
    if not records:
        raise RuntimeError("No complete RGB/segmentation/depth frame triples found.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Submit this script with qsub/qrsh on a GPU node.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Frames: {len(records)}")
    print(f"Model: {args.model}")
    print(f"CUDA: {torch.cuda.get_device_name(0)} dtype={dtype}")
    prompts = {
        "rgb": load_prompt_json(args.rgb_prompt_json, RGB_PROMPT),
        "rgb_seg_depth": load_prompt_json(args.rgb_seg_depth_prompt_json, RGB_SEG_DEPTH_PROMPT),
    }
    model = load_model(args.model, dtype=dtype, use_flash_attn=not args.no_flash_attn, load_in_8bit=args.load_in_8bit)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    for mode in args.modes:
        run_mode(args, mode, records, model, tokenizer, dtype, prompts)
    print(f"Done. Outputs: {args.output_root}")


if __name__ == "__main__":
    main()
