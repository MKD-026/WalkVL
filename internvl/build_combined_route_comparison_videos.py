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
        description="Build side-by-side RGB / model-output / RGB+Seg+Depth comparison videos."
    )
    parser.add_argument("--cropped-root", default="/projectnb/cs585/students/mkd/740/cropped_routes")
    parser.add_argument("--filtered-root", default="/projectnb/cs585/students/mkd/740/filtered")
    parser.add_argument("--seg-dir", default="/projectnb/cs585/students/mkd/740/semantic_results-2/segmented_images")
    parser.add_argument("--depth-dir", default="/projectnb/cs585/students/mkd/740/semantic_results-2/depth_maps")
    parser.add_argument("--results-root", default="/projectnb/cs585/students/mkd/740/WalkVL/experiments/internvl/outputs/filtered_walkvlm_scene_priors_r145")
    parser.add_argument("--model-logo", default="/projectnb/cs585/students/mkd/740/WalkVL/assets/logo/walkvl_logo.png")
    parser.add_argument("--output-dir", default="/projectnb/cs585/students/mkd/740/WalkVL/experiments/internvl/outputs/route_combined_comparison_videos")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--routes", nargs="+", default=["route1", "route4", "route5"])
    parser.add_argument("--route1-max-frames", type=int, default=280)
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
        return compact(strip_code_fence(record.get("response", "")))[:800]

    parts = []
    for key in (
        "spoken_guidance",
        "scene_summary",
        "people_summary",
        "spatial_summary",
        "depth_cues",
        "short_reason",
    ):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    action = payload.get("recommended_action")
    direction = payload.get("safe_direction")
    stop_now = payload.get("stop_now")
    if action or direction or stop_now is not None:
        parts.append(f"Action: {action}; direction: {direction}; stop_now: {stop_now}.")

    if not parts:
        parts.append(json.dumps(payload, ensure_ascii=False))
    return compact(" ".join(parts))[:900]


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
                outputs[key] = {
                    "summary_text": summarize_response(record),
                    "raw_response": strip_code_fence(record.get("response", "")),
                    "prompt": record.get("prompt", ""),
                    "frame_name": record.get("frame_name", f"{key}.png"),
                    "route": record.get("route"),
                }
    return outputs


def route_frames(cropped_root: Path, route_name: str, route1_max_frames: int):
    frames = list_images(cropped_root / route_name)
    if route_name == "route1" and route1_max_frames > 0:
        frames = frames[:route1_max_frames]
    return frames


def resize_cover(image, target_width, target_height):
    height, width = image.shape[:2]
    scale = max(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    x = max(0, (new_width - target_width) // 2)
    y = max(0, (new_height - target_height) // 2)
    return resized[y : y + target_height, x : x + target_width]


def resize_contain_white(image, target_width, target_height):
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2
    canvas[y : y + new_height, x : x + new_width] = resized
    return canvas


def resize_fit_no_canvas(image, max_width, max_height):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def draw_panel_title(image, title, xy, color=(255, 220, 120), scale=0.95):
    x, y = xy
    cv2.putText(image, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def draw_wrapped_text(image, text, x, y, max_width, max_lines, font_scale=0.78, line_height=34):
    chars_per_line = max(32, int(max_width / (font_scale * 17)))
    for line in textwrap.wrap(compact(text), width=chars_per_line)[:max_lines]:
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (242, 242, 242),
            1,
            cv2.LINE_AA,
        )
        y += line_height
    return y


def load_logo(path: Path, max_width: int, max_height: int):
    logo = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if logo is None:
        return None
    if logo.shape[2] == 4:
        alpha = logo[:, :, 3:4] / 255.0
        logo = (logo[:, :, :3] * alpha + np.full(logo[:, :, :3].shape, 255, dtype=np.uint8) * (1 - alpha)).astype(np.uint8)
    return resize_fit_no_canvas(logo, max_width, max_height)


def draw_output_panel(panel, logo):
    panel[:] = (255, 255, 255)
    height, width = panel.shape[:2]
    half = height // 2
    margin = 48
    cv2.line(panel, (0, half), (width, half), (218, 218, 218), 2)

    draw_panel_title(panel, "InternVL", (margin, 96), color=(32, 96, 160), scale=1.55)

    if logo is not None:
        logo_h, logo_w = logo.shape[:2]
        x = margin
        y = half + 70
        panel[y : y + logo_h, x : x + logo_w] = logo
    else:
        draw_panel_title(panel, "RGB + Segmentation + Depth", (margin, half + 96), color=(32, 96, 160), scale=1.25)


def draw_border(image, color=(35, 35, 35), thickness=1):
    height, width = image.shape[:2]
    cv2.rectangle(image, (1, 1), (width - 2, height - 2), color, thickness)


def build_grid_panel(frame_path: Path, matched_names, seg_dir: Path, depth_dir: Path, state):
    cell_w, cell_h = 256, 270
    canvas_w, canvas_h = cell_w * 5, cell_h * 4
    rgb_w, rgb_h = cell_w * 3, cell_h * 4
    side_w, side_h = cell_w * 2, cell_h * 2

    rgb = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Could not read cropped frame: {frame_path}")
    rgb = cv2.resize(rgb, (rgb_w, rgb_h), interpolation=cv2.INTER_AREA)
    panel = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    panel[0:rgb_h, 0:rgb_w] = rgb

    if frame_path.name in matched_names:
        seg_path = seg_path_for(seg_dir, frame_path.stem)
        depth_path = depth_path_for(depth_dir, frame_path.stem)
        if seg_path and depth_path:
            seg = cv2.imread(str(seg_path), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_COLOR)
            if seg is not None and depth is not None:
                state["seg"] = resize_cover(seg, side_w, side_h)
                state["depth"] = resize_cover(depth, side_w, side_h)
                state["matched_seg_depth"] += 1

    panel[0:side_h, rgb_w:canvas_w] = state["seg"]
    panel[side_h:canvas_h, rgb_w:canvas_w] = state["depth"]
    return panel


def export_route_text(route_name, frames, matched_names, rgb_outputs, mm_outputs, output_dir: Path):
    entries = []
    for frame_index, frame_path in enumerate(frames, start=1):
        if frame_path.name not in matched_names:
            continue
        rgb_record = rgb_outputs.get(frame_path.stem, {})
        mm_record = mm_outputs.get(frame_path.stem, {})
        entries.append(
            {
                "route": route_name,
                "video_frame_index_1based": frame_index,
                "frame_name": frame_path.name,
                "key": frame_path.stem,
                "rgb": rgb_record,
                "rgb_seg_depth": mm_record,
            }
        )

    json_path = output_dir / f"{route_name}_frame_outputs.json"
    txt_path = output_dir / f"{route_name}_frame_outputs.txt"
    json_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = []
    for entry in entries:
        lines.extend(
            [
                f"[{entry['route']}] video frame {entry['video_frame_index_1based']} | {entry['frame_name']}",
                "RGB:",
                entry["rgb"].get("summary_text") or entry["rgb"].get("raw_response") or "No RGB output.",
                "RGB+Seg+Depth:",
                entry["rgb_seg_depth"].get("summary_text") or entry["rgb_seg_depth"].get("raw_response") or "No RGB+Seg+Depth output.",
                "",
            ]
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, txt_path, len(entries)


def render_route(route_name, frames, matched_names, seg_dir, depth_dir, logo, out_path, fps):
    left_w, mid_w, right_w, height = 768, 1152, 1280, 1080
    writer = make_writer(out_path, fps, (left_w + mid_w + right_w, height))
    state = {
        "seg": np.full((540, 512, 3), 255, dtype=np.uint8),
        "depth": np.full((540, 512, 3), 255, dtype=np.uint8),
        "matched_seg_depth": 0,
    }
    try:
        total = len(frames)
        for frame_index, frame_path in enumerate(frames, start=1):
            rgb = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise RuntimeError(f"Could not read cropped frame: {frame_path}")
            left_panel = resize_contain_white(rgb, left_w, height)
            draw_frame_counter(left_panel, frame_index, total)
            draw_border(left_panel)

            mid_panel = np.zeros((height, mid_w, 3), dtype=np.uint8)
            draw_output_panel(mid_panel, logo)
            right_panel = build_grid_panel(frame_path, matched_names, seg_dir, depth_dir, state)
            draw_border(right_panel)

            canvas = np.full((height, left_w + mid_w + right_w, 3), 255, dtype=np.uint8)
            canvas[:, 0:left_w] = left_panel
            canvas[:, left_w : left_w + mid_w] = mid_panel
            canvas[:, left_w + mid_w :] = right_panel
            writer.write(canvas)
    finally:
        writer.release()
    return state["matched_seg_depth"]


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
    logo = load_logo(Path(args.model_logo), 440, 96)

    manifest = {}
    for route_name in args.routes:
        frames = route_frames(cropped_root, route_name, args.route1_max_frames)
        matched = filtered_names(filtered_root, route_name)
        route_out = output_dir / f"{route_name}_combined_rgb_outputs_grid_30fps.mp4"
        json_path, txt_path, output_count = export_route_text(route_name, frames, matched, rgb_outputs, mm_outputs, output_dir)
        matched_count = render_route(
            route_name,
            frames,
            matched,
            seg_dir,
            depth_dir,
            logo,
            route_out,
            args.fps,
        )
        manifest[route_name] = {
            "video": str(route_out),
            "total_video_frames": len(frames),
            "filtered_matched_frame_names": len(matched),
            "matched_frames_with_seg_depth": matched_count,
            "frame_output_records": output_count,
            "frame_outputs_json": str(json_path),
            "frame_outputs_txt": str(txt_path),
            "results_root": str(results_root),
            "layout": "3200x1080: left 2/10 RGB with frame counter, middle 4/10 model names only, right 4/10 RGB+Seg+Depth grid.",
        }
        print(f"{route_name}: frames={len(frames)} matched={len(matched)} matched_seg_depth={matched_count}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
