#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop full route frames and build RGB/context videos with segmentation/depth on matched inference frames."
    )
    parser.add_argument("--route-root", default="/projectnb/cs585/students/mkd/740")
    parser.add_argument("--cropped-root", default="/projectnb/cs585/students/mkd/740/cropped_routes")
    parser.add_argument("--filtered-root", default="/projectnb/cs585/students/mkd/740/filtered")
    parser.add_argument("--seg-dir", default="/projectnb/cs585/students/mkd/740/semantic_results-2/segmented_images")
    parser.add_argument("--depth-dir", default="/projectnb/cs585/students/mkd/740/semantic_results-2/depth_maps")
    parser.add_argument("--output-dir", default="/projectnb/cs585/students/mkd/740/route_context_videos")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--routes", nargs="+", default=["route1", "route2", "route3", "route4", "route5"])
    parser.add_argument("--skip-existing-crops", action="store_true")
    parser.add_argument("--rgb-only", action="store_true", help="Only create cropped RGB videos.")
    parser.add_argument("--grid-only", action="store_true", help="Only create grid videos.")
    return parser.parse_args()


def list_images(directory: Path):
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def crop_frame(image):
    height, width = image.shape[:2]
    cropped_width = int(round(width * 0.40))
    x_start = int(round((width - cropped_width) / 2))
    x_end = x_start + cropped_width
    return image[0:height, x_start:x_end]


def crop_route(route_dir: Path, cropped_dir: Path, skip_existing: bool):
    cropped_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in list_images(route_dir):
        out_path = cropped_dir / image_path.name
        if skip_existing and out_path.exists():
            count += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        cropped = crop_frame(image)
        cv2.imwrite(str(out_path), cropped)
        count += 1
    return count


def make_writer(path: Path, fps: float, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def resize_contain(image, target_width, target_height):
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x = (target_width - new_width) // 2
    y = (target_height - new_height) // 2
    canvas[y : y + new_height, x : x + new_width] = resized
    return canvas


def filtered_names(filtered_root: Path, route_name: str):
    route_dir = filtered_root / f"cropped_{route_name}"
    if not route_dir.exists():
        route_dir = filtered_root / route_name
    return {path.name for path in list_images(route_dir)} if route_dir.exists() else set()


def seg_path_for(seg_dir: Path, frame_stem: str):
    candidates = [
        seg_dir / f"segmented_{frame_stem}.jpg.png",
        seg_dir / f"segmented_{frame_stem}.png",
        seg_dir / f"{frame_stem}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def depth_path_for(depth_dir: Path, frame_stem: str):
    path = depth_dir / f"{frame_stem}.png"
    return path if path.exists() else None


def draw_frame_counter(image, frame_index: int, total_frames: int):
    label = f"{frame_index}/{total_frames} frames"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    margin = 16
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    x = margin
    y = image.shape[0] - margin
    cv2.rectangle(
        image,
        (x - 8, y - text_h - baseline - 8),
        (x + text_w + 8, y + baseline + 8),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(image, label, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_rgb_video(cropped_frames, out_path: Path, fps: float):
    writer = make_writer(out_path, fps, (768, 1080))
    try:
        total_frames = len(cropped_frames)
        for frame_index, frame_path in enumerate(cropped_frames, start=1):
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read cropped frame: {frame_path}")
            if image.shape[1] != 768 or image.shape[0] != 1080:
                image = cv2.resize(image, (768, 1080), interpolation=cv2.INTER_AREA)
            draw_frame_counter(image, frame_index, total_frames)
            writer.write(image)
    finally:
        writer.release()


def render_grid_video(cropped_frames, matched_names, seg_dir: Path, depth_dir: Path, out_path: Path, fps: float):
    cell_w, cell_h = 256, 270
    canvas_w, canvas_h = cell_w * 5, cell_h * 4
    rgb_w, rgb_h = cell_w * 3, cell_h * 4
    side_w, side_h = cell_w * 2, cell_h * 2
    writer = make_writer(out_path, fps, (canvas_w, canvas_h))
    matched_count = 0
    last_seg = np.zeros((side_h, side_w, 3), dtype=np.uint8)
    last_depth = np.zeros((side_h, side_w, 3), dtype=np.uint8)
    try:
        total_frames = len(cropped_frames)
        for frame_index, frame_path in enumerate(cropped_frames, start=1):
            rgb = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise RuntimeError(f"Could not read cropped frame: {frame_path}")
            if rgb.shape[1] != rgb_w or rgb.shape[0] != rgb_h:
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
                        last_seg = resize_contain(seg, side_w, side_h)
                        last_depth = resize_contain(depth, side_w, side_h)
                        matched_count += 1

            canvas[0:side_h, rgb_w:canvas_w] = last_seg
            canvas[side_h:canvas_h, rgb_w:canvas_w] = last_depth
            draw_frame_counter(canvas, frame_index, total_frames)
            writer.write(canvas)
    finally:
        writer.release()
    return matched_count


def main():
    args = parse_args()
    route_root = Path(args.route_root)
    cropped_root = Path(args.cropped_root)
    filtered_root = Path(args.filtered_root)
    seg_dir = Path(args.seg_dir)
    depth_dir = Path(args.depth_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for route_name in args.routes:
        route_dir = route_root / route_name
        cropped_dir = cropped_root / route_name
        if not route_dir.exists():
            raise FileNotFoundError(f"Route folder not found: {route_dir}")

        crop_count = crop_route(route_dir, cropped_dir, args.skip_existing_crops)
        cropped_frames = list_images(cropped_dir)
        matched = filtered_names(filtered_root, route_name)

        route_output = output_dir / route_name
        route_output.mkdir(parents=True, exist_ok=True)
        rgb_video = route_output / f"{route_name}_cropped_rgb_30fps.mp4"
        grid_video = route_output / f"{route_name}_rgb_seg_depth_grid_30fps.mp4"

        if not args.grid_only:
            render_rgb_video(cropped_frames, rgb_video, args.fps)
        matched_with_modal = None
        if not args.rgb_only:
            matched_with_modal = render_grid_video(cropped_frames, matched, seg_dir, depth_dir, grid_video, args.fps)

        manifest[route_name] = {
            "source_route_dir": str(route_dir),
            "cropped_route_dir": str(cropped_dir),
            "total_source_frames": crop_count,
            "total_cropped_frames": len(cropped_frames),
            "filtered_matched_frame_names": len(matched),
            "matched_frames_with_seg_depth": matched_with_modal,
            "rgb_video": str(rgb_video) if not args.grid_only else None,
            "grid_video": str(grid_video) if not args.rgb_only else None,
            "grid_layout": "1280x1080 canvas: RGB in left 3x4 cells, segmentation upper-right 2x2 cells, depth lower-right 2x2 cells, black elsewhere.",
        }
        print(
            f"{route_name}: cropped={len(cropped_frames)} filtered_matches={len(matched)} "
            f"matched_seg_depth={matched_with_modal}"
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
