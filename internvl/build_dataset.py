#!/usr/bin/env python3
"""
Build train/val JSONL datasets from Bench2Drive-mini for InternVL2 fine-tuning.

Outputs (in --out-dir):
  train_rgb.jsonl          RGB-only mode
  val_rgb.jsonl
  train_multimodal.jsonl   Concatenated RGB+Instance+Depth (single tall image)
  val_multimodal.jsonl
  composites/<scenario>/<frame>.png   saved composite images

Usage:
  python build_dataset.py
  python build_dataset.py --data-dir /path/to/Bench2Drive-mini --out-dir data/
"""

import gzip, json, math, argparse
from pathlib import Path
from PIL import Image
import numpy as np

# ── Navigation command codes (CARLA) ────────────────────────────────────────
COMMAND_MAP = {
    1: 'turn left', 2: 'turn right', 3: 'go straight',
    4: 'follow lane', 5: 'change lane left', 6: 'change lane right',
}

# ── Val scenarios (held out by scenario to avoid leakage) ───────────────────
VAL_SCENARIOS = {
    'DynamicObjectCrossing_Town02_Route13_Weather6',
    'YieldToEmergencyVehicle_Town04_Route165_Weather7',
}

# ── Defaults ────────────────────────────────────────────────────────────────
WALKVL_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = '/projectnb/cs585/students/mkd/740/Bench2Drive/Bench2Drive-mini'
OUT_DIR = str(WALKVL_ROOT / 'experiments' / 'internvl' / 'data')
PROMPT_DIR = str(WALKVL_ROOT / 'internvl' / 'prompts')


# ── Ground-truth helpers ─────────────────────────────────────────────────────

def load_anno(path: Path) -> dict:
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def waypoint_bearing(anno: dict) -> float:
    """Relative bearing in degrees from ego heading to near waypoint (+ve = right)."""
    dx = anno['x_command_near'] - anno['x']
    dy = anno['y_command_near'] - anno['y']
    rel = (math.atan2(dy, dx) - anno['theta'] + math.pi) % (2 * math.pi) - math.pi
    return math.degrees(rel)


def route_context(anno: dict) -> str:
    speed = anno.get('speed', 0.0)
    cmd   = COMMAND_MAP.get(anno.get('command_near', 4), 'follow lane')
    bear  = waypoint_bearing(anno)
    return f'Route: speed={speed:.1f} m/s | command={cmd} | near-waypoint bearing={bear:+.1f}°'


def compute_safe_direction(anno: dict) -> str:
    if anno.get('should_brake', False):
        return 'stop'
    ego_x, ego_y = anno['x'], anno['y']
    theta = anno['theta']
    wp_x, wp_y = anno['x_command_near'], anno['y_command_near']
    dx, dy = wp_x - ego_x, wp_y - ego_y
    wp_angle = math.atan2(dy, dx)
    rel = (wp_angle - theta + math.pi) % (2 * math.pi) - math.pi
    if abs(rel) < math.radians(25):
        return 'forward'
    return 'left' if rel < 0 else 'right'


def ego_distance(anno: dict, loc: list) -> float:
    return math.sqrt((loc[0] - anno['x']) ** 2 + (loc[1] - anno['y']) ** 2)


def distance_label(d: float) -> str:
    if d < 8:   return 'near'
    if d < 20:  return 'mid'
    return 'far'


def risk_from_distance(d: str) -> str:
    return {'near': 'high', 'mid': 'med', 'far': 'low'}[d]


def extract_pedestrians(anno: dict, mode: str) -> list:
    peds = []
    for bb in anno.get('bounding_boxes', []):
        is_walker = (bb.get('base_type') == 'walker' or
                     'pedestrian' in bb.get('class', '').lower())
        if not is_walker:
            continue
        d = ego_distance(anno, bb['location'])
        dlabel = distance_label(d)
        entry = {
            'risk': risk_from_distance(dlabel),
            'crossing_intent': round(0.65 if dlabel == 'near' else 0.3, 2),
        }
        if mode == 'multimodal':
            entry['distance'] = dlabel
        peds.append(entry)
    return peds


def infer_safe_zones(instance_path: Path) -> list:
    arr = np.array(Image.open(instance_path))
    r = arr[:, :, 0]
    h, w = r.shape
    zones = []
    if np.any(r[:, :w // 2] == 8):  zones.append('sidewalk_left')
    if np.any(r[:, w // 2:] == 8):  zones.append('sidewalk_right')
    if np.any(r == 6):               zones.append('roadline_visible')
    if np.any(r == 14):              zones.append('ground_ahead')
    return zones or ['road_only']


def build_answer(anno: dict, safe_zones: list, mode: str) -> str:
    peds      = extract_pedestrians(anno, mode)
    safe_dir  = compute_safe_direction(anno)
    stop_now  = bool(anno.get('should_brake', False))

    parts = []
    if stop_now:
        parts.append('braking required')
    high = [p for p in peds if p['risk'] == 'high']
    if high:
        parts.append(f'{len(high)} high-risk pedestrian(s) nearby')
    parts.append(f'navigating {safe_dir}')

    return json.dumps({
        'pedestrians':   peds,
        'safe_zones':    safe_zones,
        'safe_direction': safe_dir,
        'stop_now':      stop_now,
        'uncertainty':   0.05,
        'short_reason':  ', '.join(parts) + '.',
    })


# ── Image helpers ─────────────────────────────────────────────────────────────

def make_composite(rgb_path: Path, inst_path: Path, dep_path: Path) -> Image.Image:
    """Stack RGB / Instance / Depth vertically into a single tall image."""
    rgb  = Image.open(rgb_path).convert('RGB')
    inst = Image.open(inst_path).convert('RGB')
    dep  = Image.open(dep_path).convert('L').convert('RGB')  # grayscale → RGB
    w, h = rgb.size
    canvas = Image.new('RGB', (w, h * 3))
    canvas.paste(rgb,  (0, 0))
    canvas.paste(inst, (0, h))
    canvas.paste(dep,  (0, h * 2))
    return canvas


# ── Per-scenario processing ───────────────────────────────────────────────────

def process_scenario(scenario_dir: Path, prompt_rgb: str, prompt_mm: str,
                     composite_dir: Path):
    anno_dir = scenario_dir / 'anno'
    rgb_dir  = scenario_dir / 'camera' / 'rgb_front'
    inst_dir = scenario_dir / 'camera' / 'instance_front'
    dep_dir  = scenario_dir / 'camera' / 'depth_front'

    rgb_records, mm_records = [], []

    for anno_file in sorted(anno_dir.glob('*.json.gz')):
        frame = anno_file.stem.replace('.json', '')       # e.g. '00100'
        rgb_p  = rgb_dir  / f'{frame}.jpg'
        inst_p = inst_dir / f'{frame}.png'
        dep_p  = dep_dir  / f'{frame}.png'
        if not (rgb_p.exists() and inst_p.exists() and dep_p.exists()):
            continue

        anno       = load_anno(anno_file)
        safe_zones = infer_safe_zones(inst_p)
        route      = route_context(anno)

        # ── RGB record ───────────────────────────────────────────────────────
        rgb_records.append({
            'id':    f'{scenario_dir.name}_{frame}_rgb',
            'image': str(rgb_p),
            'conversations': [
                {'from': 'human', 'value': f'<image>\n{route}\n\n{prompt_rgb}'},
                {'from': 'gpt',   'value': build_answer(anno, safe_zones, 'rgb')},
            ],
        })

        # ── Multimodal (composite) record ────────────────────────────────────
        comp_path = composite_dir / scenario_dir.name / f'{frame}.png'
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        if not comp_path.exists():
            make_composite(rgb_p, inst_p, dep_p).save(comp_path)

        mm_records.append({
            'id':    f'{scenario_dir.name}_{frame}_mm',
            'image': str(comp_path),
            'conversations': [
                {'from': 'human', 'value': f'<image>\n{route}\n\n{prompt_mm}'},
                {'from': 'gpt',   'value': build_answer(anno, safe_zones, 'multimodal')},
            ],
        })

    return rgb_records, mm_records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',   default=DATA_DIR)
    parser.add_argument('--out-dir',    default=OUT_DIR)
    parser.add_argument('--prompt-dir', default=PROMPT_DIR)
    args = parser.parse_args()

    data_dir      = Path(args.data_dir)
    out_dir       = Path(args.out_dir)
    composite_dir = out_dir / 'composites'
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(args.prompt_dir) / 'ped_nav_rgb_only_prompt.json') as f:
        prompt_rgb = json.load(f)['question']
    with open(Path(args.prompt_dir) / 'ped_nav_prompt.json') as f:
        prompt_mm = json.load(f)['question']

    splits = {'train_rgb': [], 'val_rgb': [], 'train_multimodal': [], 'val_multimodal': []}

    for scenario_dir in sorted(data_dir.iterdir()):
        if not scenario_dir.is_dir() or not (scenario_dir / 'anno').exists():
            continue
        print(f'Processing {scenario_dir.name} ...', flush=True)
        rgb_recs, mm_recs = process_scenario(scenario_dir, prompt_rgb, prompt_mm, composite_dir)
        prefix = 'val' if scenario_dir.name in VAL_SCENARIOS else 'train'
        splits[f'{prefix}_rgb'].extend(rgb_recs)
        splits[f'{prefix}_multimodal'].extend(mm_recs)

    for name, records in splits.items():
        out_path = out_dir / f'{name}.jsonl'
        with open(out_path, 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + '\n')
        print(f'  {name}: {len(records):,} records → {out_path}')


if __name__ == '__main__':
    main()
