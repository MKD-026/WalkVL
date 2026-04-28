#!/usr/bin/env python3
import argparse
import json
import re
import textwrap
from pathlib import Path

import cv2
import numpy as np

from build_route_context_videos import (
    depth_path_for,
    draw_frame_counter,
    filtered_names,
    list_images,
    make_writer,
    resize_contain,
    seg_path_for,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build route videos with persisted segmentation/depth and model-output captions."
    )
    parser.add_argument("--cropped-root", default="/projectnb/cs585/students/mkd/740/cropped_routes")
    parser.add_argument("--filtered-root", default="/projectnb/cs585/students/mkd/740/filtered")
    parser.add_argument("--seg-dir", default="/projectnb/cs585/students/mkd/740/semantic_results-2/segmented_images")
    parser.add_argument("--depth-dir", default="/projectnb/cs585/students/mkd/740/semantic_results-2/depth_maps")
    parser.add_argument("--results-root", default="/projectnb/cs585/students/mkd/740/WalkVL/experiments/internvl/outputs/filtered_walkvlm_spatial_prompts")
    parser.add_argument("--output-dir", default="/projectnb/cs585/students/mkd/740/WalkVL/experiments/internvl/outputs/route_context_videos_with_outputs")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--routes", nargs="+", default=["route1", "route2", "route3", "route4", "route5"])
    parser.add_argument("--route1-max-frames", type=int, default=280)
    parser.add_argument("--caption-height", type=int, default=210)
    return parser.parse_args()


def strip_code_fence(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def extract_json(text):
    cleaned = strip_code_fence(text)
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


def compact(text):
    return " ".join(str(text or "").split())


def summarize_response(record):
    payload = extract_json(record.get("response", ""))
    if not isinstance(payload, dict):
        return compact(strip_code_fence(record.get("response", "")))[:500]

    parts = []
    for key in ("spoken_guidance", "people_summary", "spatial_summary", "depth_cues", "short_reason"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    action = payload.get("recommended_action")
    direction = payload.get("safe_direction")
    stop_now = payload.get("stop_now")
    if action or direction or stop_now is not None:
        parts.append(f"Action: {action}; direction: {direction}; stop_now: {stop_now}.")
    if not parts and isinstance(payload.get("scene_summary"), str):
        parts.append(payload["scene_summary"])
    return compact(" ".join(parts))[:650]


def load_results(results_path: Path):
    outputs = {}
    if not results_path.exists():
        return outputs
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("key") or Path(record.get("frame_name", "")).stem
            if key:
                outputs[key] = summarize_response(record)
    return outputs


def draw_caption(image, title, text, caption_height):
    if not text:
        text = "No model output yet."
    h, w = image.shape[:2]
    top = max(0, h - caption_height)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, top), (w, h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, dst=image)

    cv2.putText(
        image,
        title,
        (16, top + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 220, 120),
        2,
        cv2.LINE_AA,
    )
    y = top + 68
    chars_per_line = max(42, w // 16)
    for line in textwrap.wrap(compact(text), width=chars_per_line)[:6]:
        cv2.putText(
            image,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        y += 28


def route_frames(cropped_root: Path, route_name: str, route1_max_frames: int):
    frames = list_images(cropped_root / route_name)
    if route_name == "route1" and route1_max_frames > 0:
        frames = frames[:route1_max_frames]
    return frames


def render_rgb_video(cropped_frames, matched_names, rgb_outputs, out_path: Path, fps: float, caption_height: int):
    writer = make_writer(out_path, fps, (768, 1080))
    latest_text = ""
    try:
        total = len(cropped_frames)
        for idx, frame_path in enumerate(cropped_frames, start=1):
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read cropped frame: {frame_path}")
            image = cv2.resize(image, (768, 1080), interpolation=cv2.INTER_AREA)
            if frame_path.name in matched_names and frame_path.stem in rgb_outputs:
                latest_text = rgb_outputs[frame_path.stem]
            draw_caption(image, "RGB model output", latest_text, caption_height)
            draw_frame_counter(image, idx, total)
            writer.write(image)
    finally:
        writer.release()


def render_grid_video(
    cropped_frames,
    matched_names,
    mm_outputs,
    seg_dir: Path,
    depth_dir: Path,
    out_path: Path,
    fps: float,
    caption_height: int,
):
    cell_w, cell_h = 256, 270
    canvas_w, canvas_h = cell_w * 5, cell_h * 4
    rgb_w, rgb_h = cell_w * 3, cell_h * 4
    side_w, side_h = cell_w * 2, cell_h * 2
    writer = make_writer(out_path, fps, (canvas_w, canvas_h))
    latest_seg = np.zeros((side_h, side_w, 3), dtype=np.uint8)
    latest_depth = np.zeros((side_h, side_w, 3), dtype=np.uint8)
    latest_text = ""
    matched_count = 0
    try:
        total = len(cropped_frames)
        for idx, frame_path in enumerate(cropped_frames, start=1):
            rgb = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise RuntimeError(f"Could not read cropped frame: {frame_path}")
            rgb = cv2.resize(rgb, (rgb_w, rgb_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            canvas[0:rgb_h, 0:rgb_w] = rgb

            if frame_path.name in matched_names:
                seg_path = seg_path_for(seg_dir, frame_path.stem)
                depth_path = depth_path_for(depth_dir, frame_path.stem)
                if seg_path and depth_path:
                    seg = cv2.imread(str(seg_path), cv2.IMREAD_COLOR)
                    depth = cv2.imread(str(depth_path), cv2.IMREAD_COLOR)
                    if seg is not None and depth is not None:
                        latest_seg = resize_contain(seg, side_w, side_h)
                        latest_depth = resize_contain(depth, side_w, side_h)
                        matched_count += 1
                if frame_path.stem in mm_outputs:
                    latest_text = mm_outputs[frame_path.stem]

            canvas[0:side_h, rgb_w:canvas_w] = latest_seg
            canvas[side_h:canvas_h, rgb_w:canvas_w] = latest_depth
            draw_caption(canvas, "RGB+Seg+Depth model output", latest_text, caption_height)
            draw_frame_counter(canvas, idx, total)
            writer.write(canvas)
    finally:
        writer.release()
    return matched_count


def main():
    args = parse_args()
    cropped_root = Path(args.cropped_root)
    filtered_root = Path(args.filtered_root)
    seg_dir = Path(args.seg_dir)
    depth_dir = Path(args.depth_dir)
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_outputs = load_results(results_root / "rgb" / "results.jsonl")
    mm_outputs = load_results(results_root / "rgb_seg_depth" / "results.jsonl")

    manifest = {}
    for route_name in args.routes:
        frames = route_frames(cropped_root, route_name, args.route1_max_frames)
        matched = filtered_names(filtered_root, route_name)
        route_out = output_dir / route_name
        route_out.mkdir(parents=True, exist_ok=True)
        rgb_video = route_out / f"{route_name}_rgb_with_model_outputs_30fps.mp4"
        grid_video = route_out / f"{route_name}_grid_with_model_outputs_30fps.mp4"

        render_rgb_video(frames, matched, rgb_outputs, rgb_video, args.fps, args.caption_height)
        matched_count = render_grid_video(
            frames,
            matched,
            mm_outputs,
            seg_dir,
            depth_dir,
            grid_video,
            args.fps,
            args.caption_height,
        )
        manifest[route_name] = {
            "total_video_frames": len(frames),
            "filtered_matched_frame_names": len(matched),
            "matched_frames_with_seg_depth": matched_count,
            "rgb_video": str(rgb_video),
            "grid_video": str(grid_video),
            "results_root": str(results_root),
            "caption_behavior": "Latest available model output persists until the next matched inference frame.",
        }
        print(f"{route_name}: frames={len(frames)} matched={len(matched)} matched_seg_depth={matched_count}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
