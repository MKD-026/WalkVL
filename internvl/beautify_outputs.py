#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "internvl"


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clean_text(text, limit=700):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def pct(value):
    return f"{100 * float(value):.1f}%"


def title_from_path(path):
    return Path(path).name.replace("_", " ").replace("-", " ").title()


def report_stem(path, base):
    rel = Path(path).parent.relative_to(base)
    return "__".join(rel.parts).replace(" ", "_")


def summarize_analysis(path):
    data = load_json(path)
    lines = [f"Analysis: {title_from_path(Path(path).parent)}", ""]
    total = data.get("total_frames", 0)
    lines.append(f"Frames evaluated: {total}")
    if "json_parse_success_frames" in data:
        ok = data.get("json_parse_success_frames", 0)
        rate = data.get("json_parse_success_rate", ok / max(total, 1))
        lines.append(f"Structured JSON responses: {ok} ({pct(rate)})")
    if data.get("avg_latency_s") is not None:
        lines.append(f"Average latency: {data['avg_latency_s']:.2f}s/frame")
    if data.get("mean_uncertainty") is not None:
        lines.append(f"Mean uncertainty: {data['mean_uncertainty']:.3f}")
    for key in (
        "recommended_action_counts",
        "safe_direction_counts",
        "pedestrian_risk_counts",
    ):
        if data.get(key):
            pretty = key.replace("_", " ").title()
            lines.append(f"{pretty}: {data[key]}")
    if "stop_now_true_frames" in data:
        lines.append(f"Stop-now frames: {data['stop_now_true_frames']}")
    return "\n".join(lines).strip() + "\n"


def summarize_comparison(path, max_examples=8):
    data = load_json(path)
    summary = data.get("summary", {})
    comparisons = data.get("comparisons", [])
    lines = ["RGB vs RGB+Seg+Depth Comparison", ""]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"{key.replace('_', ' ').title()}: {value:.3f}")
        else:
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
    lines.append("")
    lines.append("Notable Frame Differences")
    interesting = [
        item
        for item in comparisons
        if any(item.get("changed", {}).values())
        or item.get("rgb", {}).get("json_ok") != item.get("rgb_seg_depth", {}).get("json_ok")
    ]
    for item in interesting[:max_examples]:
        rgb = item.get("rgb", {})
        mm = item.get("rgb_seg_depth", {})
        lines.append("")
        lines.append(f"Frame: {item.get('key')}")
        lines.append(
            "RGB: "
            f"json={rgb.get('json_ok')} | action={rgb.get('recommended_action')} | "
            f"direction={rgb.get('safe_direction')} | stop={rgb.get('stop_now')}"
        )
        lines.append(
            "RGB+Seg+Depth: "
            f"json={mm.get('json_ok')} | action={mm.get('recommended_action')} | "
            f"direction={mm.get('safe_direction')} | stop={mm.get('stop_now')}"
        )
        lines.append(f"Changed fields: {item.get('changed')}")
        lines.append(f"RGB response: {clean_text(rgb.get('short_reason') or rgb.get('raw_response'))}")
        lines.append(
            "RGB+Seg+Depth response: "
            f"{clean_text(mm.get('short_reason') or mm.get('raw_response'))}"
        )
    return "\n".join(lines).strip() + "\n"


def summarize_results(path, max_examples=12):
    records = load_jsonl(path)
    parsed = 0
    errors = 0
    latency = []
    directions = Counter()
    actions = Counter()
    examples = []

    for rec in records:
        errors += int(bool(rec.get("error")))
        if isinstance(rec.get("latency_s"), (int, float)):
            latency.append(float(rec["latency_s"]))
        payload = extract_json(rec.get("response", ""))
        if isinstance(payload, dict):
            parsed += 1
            if isinstance(payload.get("safe_direction"), str):
                directions[payload["safe_direction"]] += 1
            if isinstance(payload.get("recommended_action"), str):
                actions[payload["recommended_action"]] += 1
        if len(examples) < max_examples:
            examples.append((rec, payload))

    lines = [f"Frame Results: {title_from_path(Path(path).parent)}", ""]
    lines.append(f"Frames: {len(records)}")
    lines.append(f"Errors: {errors}")
    lines.append(f"Structured JSON responses: {parsed} ({pct(parsed / max(len(records), 1))})")
    if latency:
        lines.append(f"Average latency: {sum(latency) / len(latency):.2f}s/frame")
    if actions:
        lines.append(f"Recommended actions: {dict(actions)}")
    if directions:
        lines.append(f"Safe directions: {dict(directions)}")
    lines.append("")
    lines.append("Sample Responses")
    for rec, payload in examples:
        text = ""
        if isinstance(payload, dict):
            text = payload.get("short_reason") or payload.get("scene_summary") or rec.get("response", "")
        else:
            text = rec.get("response", "")
        lines.append("")
        lines.append(f"{rec.get('frame_name') or rec.get('idx')}: {clean_text(text)}")
    return "\n".join(lines).strip() + "\n"


def extract_json(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def summarize_metrics(path):
    records = load_jsonl(path)
    train = [r for r in records if r.get("event") == "train_step"]
    val = [r for r in records if r.get("event") == "val_epoch"]
    best = [r for r in records if r.get("event") == "best_checkpoint"]
    lines = [f"Training Metrics: {Path(path).parent.name}", ""]
    lines.append(f"Train steps logged: {len(train)}")
    lines.append(f"Validation epochs logged: {len(val)}")
    if train:
        lines.append(f"First train loss: {train[0].get('train_loss')}")
        lines.append(f"Last train loss: {train[-1].get('train_loss')}")
    if val:
        lines.append(f"First val loss: {val[0].get('val_loss')}")
        lines.append(f"Last val loss: {val[-1].get('val_loss')}")
        min_val = min(v.get("val_loss", float("inf")) for v in val)
        lines.append(f"Best val loss seen: {min_val}")
    if best:
        last_best = best[-1]
        lines.append(
            f"Best checkpoint event: epoch={last_best.get('epoch')} val_loss={last_best.get('val_loss')}"
        )
    return "\n".join(lines).strip() + "\n"


def summarize_logs(log_dir):
    lines = ["Training Log Files", ""]
    for path in sorted(Path(log_dir).glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        status = "completed" if "Training complete" in text else "check log"
        losses = re.findall(r"val_loss=([0-9.]+)", text)
        lines.append(f"{path.name}: {status}")
        if losses:
            lines.append(f"  Last val_loss: {losses[-1]}")
    return "\n".join(lines).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Create final readable WalkVL InternVL TXT reports.")
    parser.add_argument("--exp-root", default=str(EXP))
    parser.add_argument("--out-dir", default=str(EXP / "final_txt"))
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    exp = Path(args.exp_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = exp / "outputs"
    written = []

    for analysis in sorted(outputs.glob("**/analysis.json")):
        target = out_dir / f"{report_stem(analysis, outputs)}__analysis.txt"
        target.write_text(summarize_analysis(analysis), encoding="utf-8")
        written.append(target)

    for results in sorted(outputs.glob("**/results.jsonl")):
        target = out_dir / f"{report_stem(results, outputs)}__frame_results.txt"
        target.write_text(summarize_results(results, args.max_examples), encoding="utf-8")
        written.append(target)

    for comparison in sorted(outputs.glob("**/response_comparison.json")):
        target = out_dir / f"{report_stem(comparison, outputs)}__rgb_vs_multimodal.txt"
        target.write_text(summarize_comparison(comparison, args.max_examples), encoding="utf-8")
        written.append(target)

    for metrics in sorted((exp / "checkpoints").glob("*/metrics.jsonl")):
        target = out_dir / f"{metrics.parent.name}_training_metrics.txt"
        target.write_text(summarize_metrics(metrics), encoding="utf-8")
        written.append(target)

    log_dir = exp / "logs"
    if log_dir.exists():
        target = out_dir / "training_logs_summary.txt"
        target.write_text(summarize_logs(log_dir), encoding="utf-8")
        written.append(target)

    manifest = out_dir / "MANIFEST.txt"
    manifest_entries = [
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        for path in written
    ]
    manifest.write_text(
        "WalkVL InternVL final text reports\n\n"
        + "\n".join(manifest_entries)
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(written)} report files to {out_dir}")


if __name__ == "__main__":
    main()
