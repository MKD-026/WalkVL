#!/usr/bin/env python3
import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


PROXIMITY_TERMS = {
    "near",
    "nearby",
    "close",
    "closer",
    "closest",
    "far",
    "farther",
    "distant",
    "mid",
    "distance",
    "depth",
}

SEMANTIC_TERMS = {
    "curb",
    "sidewalk",
    "crosswalk",
    "walkable",
    "tree",
    "road",
    "pedestrian",
    "vehicle",
    "building",
    "stairs",
    "obstacle",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare RGB vs RGB+Seg+Depth WalkVLM-like responses.")
    parser.add_argument("--rgb", required=True, help="RGB results.jsonl")
    parser.add_argument("--rgb-seg-depth", required=True, help="RGB+Seg+Depth results.jsonl")
    parser.add_argument("--output-json", required=True, help="Path to comparison JSON")
    parser.add_argument("--output-md", required=True, help="Path to readable Markdown report")
    parser.add_argument("--max-response-chars", type=int, default=700)
    return parser.parse_args()


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def key(record):
    return str(record.get("key") or record.get("frame_name") or record.get("id") or record.get("idx"))


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


def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def response_summary(record):
    payload = extract_json(record.get("response", ""))
    summary = {
        "json_ok": isinstance(payload, dict),
        "recommended_action": None,
        "safe_direction": None,
        "stop_now": None,
        "short_reason": None,
        "risk_levels": [],
        "raw_response": strip_code_fence(record.get("response", "")),
    }
    if not isinstance(payload, dict):
        return summary

    summary["recommended_action"] = payload.get("recommended_action")
    summary["safe_direction"] = payload.get("safe_direction")
    summary["stop_now"] = payload.get("stop_now")
    summary["short_reason"] = payload.get("short_reason") or payload.get("scene_summary")

    pedestrians = payload.get("pedestrians")
    if isinstance(pedestrians, list):
        for ped in pedestrians:
            if isinstance(ped, dict):
                risk = ped.get("risk") or ped.get("risk_level")
                if isinstance(risk, str):
                    summary["risk_levels"].append(risk)
    return summary


def term_hits(text, terms):
    normalized = normalize_text(text)
    tokens = set(normalized.split())
    return sorted(term for term in terms if term in tokens)


def truncate(text, limit):
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def compare_records(rgb_record, mm_record):
    rgb = response_summary(rgb_record)
    mm = response_summary(mm_record)
    rgb_text = rgb["short_reason"] or rgb["raw_response"]
    mm_text = mm["short_reason"] or mm["raw_response"]
    return {
        "key": key(rgb_record),
        "rgb": rgb,
        "rgb_seg_depth": mm,
        "changed": {
            "recommended_action": rgb["recommended_action"] != mm["recommended_action"],
            "safe_direction": rgb["safe_direction"] != mm["safe_direction"],
            "stop_now": rgb["stop_now"] != mm["stop_now"],
            "risk_levels": rgb["risk_levels"] != mm["risk_levels"],
        },
        "text_similarity": SequenceMatcher(None, normalize_text(rgb_text), normalize_text(mm_text)).ratio(),
        "rgb_proximity_terms": term_hits(rgb_text, PROXIMITY_TERMS),
        "rgb_seg_depth_proximity_terms": term_hits(mm_text, PROXIMITY_TERMS),
        "rgb_semantic_terms": term_hits(rgb_text, SEMANTIC_TERMS),
        "rgb_seg_depth_semantic_terms": term_hits(mm_text, SEMANTIC_TERMS),
        "latency_s": {
            "rgb": rgb_record.get("latency_s"),
            "rgb_seg_depth": mm_record.get("latency_s"),
        },
    }


def summarize(comparisons):
    total = len(comparisons)
    if total == 0:
        return {}
    return {
        "paired_frames": total,
        "rgb_json_ok": sum(item["rgb"]["json_ok"] for item in comparisons),
        "rgb_seg_depth_json_ok": sum(item["rgb_seg_depth"]["json_ok"] for item in comparisons),
        "recommended_action_changes": sum(item["changed"]["recommended_action"] for item in comparisons),
        "safe_direction_changes": sum(item["changed"]["safe_direction"] for item in comparisons),
        "stop_now_changes": sum(item["changed"]["stop_now"] for item in comparisons),
        "risk_level_changes": sum(item["changed"]["risk_levels"] for item in comparisons),
        "mean_text_similarity": sum(item["text_similarity"] for item in comparisons) / total,
        "rgb_proximity_frames": sum(bool(item["rgb_proximity_terms"]) for item in comparisons),
        "rgb_seg_depth_proximity_frames": sum(bool(item["rgb_seg_depth_proximity_terms"]) for item in comparisons),
        "rgb_semantic_frames": sum(bool(item["rgb_semantic_terms"]) for item in comparisons),
        "rgb_seg_depth_semantic_frames": sum(bool(item["rgb_seg_depth_semantic_terms"]) for item in comparisons),
    }


def write_markdown(path, summary, comparisons, max_response_chars):
    lines = ["# RGB vs RGB+Seg+Depth Response Comparison", ""]
    lines.append("## Summary")
    for name, value in summary.items():
        lines.append(f"- **{name}**: {value}")
    lines.append("")
    lines.append("## Frame Comparisons")
    for item in comparisons:
        rgb = item["rgb"]
        mm = item["rgb_seg_depth"]
        lines.append(f"### {item['key']}")
        lines.append(
            f"- **RGB**: action={rgb['recommended_action']} | direction={rgb['safe_direction']} | stop={rgb['stop_now']} | risks={rgb['risk_levels']} | json={rgb['json_ok']}"
        )
        lines.append(
            f"- **RGB+Seg+Depth**: action={mm['recommended_action']} | direction={mm['safe_direction']} | stop={mm['stop_now']} | risks={mm['risk_levels']} | json={mm['json_ok']}"
        )
        lines.append(f"- **Changes**: {item['changed']} | text_similarity={item['text_similarity']:.3f}")
        lines.append(f"- **Term hits**: RGB proximity={item['rgb_proximity_terms']}, MM proximity={item['rgb_seg_depth_proximity_terms']}")
        lines.append(f"- **RGB response**: {truncate(rgb['short_reason'] or rgb['raw_response'], max_response_chars)}")
        lines.append(f"- **RGB+Seg+Depth response**: {truncate(mm['short_reason'] or mm['raw_response'], max_response_chars)}")
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    rgb_records = {key(record): record for record in load_jsonl(args.rgb)}
    mm_records = {key(record): record for record in load_jsonl(args.rgb_seg_depth)}
    shared_keys = sorted(set(rgb_records) & set(mm_records))
    comparisons = [compare_records(rgb_records[item_key], mm_records[item_key]) for item_key in shared_keys]
    output = {
        "summary": summarize(comparisons),
        "comparisons": comparisons,
        "missing_from_rgb": sorted(set(mm_records) - set(rgb_records)),
        "missing_from_rgb_seg_depth": sorted(set(rgb_records) - set(mm_records)),
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, output["summary"], comparisons, args.max_response_chars)
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
