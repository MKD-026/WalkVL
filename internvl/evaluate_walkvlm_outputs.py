#!/usr/bin/env python3
import argparse
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


ACTION_VALUES = {"move_forward", "turn_left", "turn_right", "slow_down", "stop"}
SAFE_DIRECTION_VALUES = {"left", "right", "forward", "stop"}
RISK_VALUES = {"low", "med", "medium", "high"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate WalkVLM-like InternVL outputs with optional references."
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more results.jsonl files.",
    )
    parser.add_argument(
        "--references",
        default=None,
        help=(
            "Optional JSONL references keyed by key/frame_name/id. Supported fields: "
            "answer, reminder, short_reason, recommended_action, safe_direction, stop_now, risk."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for combined evaluation JSON.",
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


def tokenize(text):
    return normalize_text(text).split()


def token_f1(pred, ref):
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_len(a, b):
    rows = len(a) + 1
    cols = len(b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i, ai in enumerate(a, start=1):
        for j, bj in enumerate(b, start=1):
            if ai == bj:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(pred, ref):
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def modified_precision(pred_tokens, ref_tokens, n):
    pred_ngrams = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
    ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
    if not pred_ngrams:
        return 0.0
    return sum((pred_ngrams & ref_ngrams).values()) / sum(pred_ngrams.values())


def bleu4(pred, ref):
    pred_tokens = tokenize(pred)
    ref_tokens = tokenize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    precisions = [modified_precision(pred_tokens, ref_tokens, n) for n in range(1, 5)]
    smooth = 1e-9
    geo_mean = math.exp(sum(math.log(max(p, smooth)) for p in precisions) / 4)
    brevity = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))
    return brevity * geo_mean


def response_text(payload, raw_response):
    if isinstance(payload, dict):
        for key in ("short_reason", "reminder", "scene_summary", "depth_cues"):
            if isinstance(payload.get(key), str) and payload[key].strip():
                return payload[key]
    return strip_code_fence(raw_response)


def record_key(record):
    return str(record.get("key") or record.get("frame_name") or record.get("id") or record.get("idx"))


def load_references(path):
    if not path:
        return {}
    refs = {}
    for rec in load_jsonl(path):
        refs[record_key(rec)] = rec
    return refs


def normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def get_risks(payload):
    risks = []
    if not isinstance(payload, dict):
        return risks
    pedestrians = payload.get("pedestrians")
    if isinstance(pedestrians, list):
        for ped in pedestrians:
            if isinstance(ped, dict):
                risk = ped.get("risk", ped.get("risk_level"))
                if isinstance(risk, str):
                    risks.append(risk.lower())
    return risks


def evaluate_file(path, references):
    records = load_jsonl(path)
    parsed_payloads = []
    latencies = []
    errors = 0
    actions = Counter()
    safe_directions = Counter()
    risks = Counter()
    invalid_actions = Counter()
    invalid_directions = Counter()
    stop_true = 0
    uncertainty_values = []
    texts = []
    reference_scores = {
        "action_correct": 0,
        "action_total": 0,
        "safe_direction_correct": 0,
        "safe_direction_total": 0,
        "stop_now_correct": 0,
        "stop_now_total": 0,
        "risk_correct": 0,
        "risk_total": 0,
        "answer_exact_correct": 0,
        "answer_exact_total": 0,
        "token_f1": [],
        "rouge_l": [],
        "bleu4": [],
    }

    for rec in records:
        if rec.get("error"):
            errors += 1
        if isinstance(rec.get("latency_s"), (int, float)):
            latencies.append(float(rec["latency_s"]))
        payload = extract_json(rec.get("response", ""))
        parsed_payloads.append(payload)

        if isinstance(payload, dict):
            action = payload.get("recommended_action")
            if isinstance(action, str):
                actions[action] += 1
                if action not in ACTION_VALUES:
                    invalid_actions[action] += 1
            direction = payload.get("safe_direction")
            if isinstance(direction, str):
                safe_directions[direction] += 1
                if direction not in SAFE_DIRECTION_VALUES:
                    invalid_directions[direction] += 1
            if normalize_bool(payload.get("stop_now")) is True:
                stop_true += 1
            if isinstance(payload.get("uncertainty"), (int, float)):
                uncertainty_values.append(float(payload["uncertainty"]))
            for risk in get_risks(payload):
                risks[risk] += 1

        text = response_text(payload, rec.get("response", ""))
        texts.append(text)

        ref = references.get(record_key(rec))
        if ref and isinstance(payload, dict):
            if "recommended_action" in ref:
                reference_scores["action_total"] += 1
                reference_scores["action_correct"] += int(payload.get("recommended_action") == ref.get("recommended_action"))
            if "safe_direction" in ref:
                reference_scores["safe_direction_total"] += 1
                reference_scores["safe_direction_correct"] += int(payload.get("safe_direction") == ref.get("safe_direction"))
            if "stop_now" in ref:
                reference_scores["stop_now_total"] += 1
                reference_scores["stop_now_correct"] += int(normalize_bool(payload.get("stop_now")) == normalize_bool(ref.get("stop_now")))
            if "risk" in ref:
                pred_risks = get_risks(payload)
                reference_scores["risk_total"] += 1
                reference_scores["risk_correct"] += int(str(ref["risk"]).lower() in pred_risks)

            ref_text = ref.get("answer") or ref.get("reminder") or ref.get("short_reason")
            if ref_text:
                reference_scores["answer_exact_total"] += 1
                reference_scores["answer_exact_correct"] += int(normalize_text(text) == normalize_text(ref_text))
                reference_scores["token_f1"].append(token_f1(text, ref_text))
                reference_scores["rouge_l"].append(rouge_l_f1(text, ref_text))
                reference_scores["bleu4"].append(bleu4(text, ref_text))

    adjacent_similarities = [
        SequenceMatcher(None, normalize_text(texts[i - 1]), normalize_text(texts[i])).ratio()
        for i in range(1, len(texts))
    ]
    exact_repeat_count = sum(
        normalize_text(texts[i - 1]) == normalize_text(texts[i])
        for i in range(1, len(texts))
    )

    def mean(values):
        return sum(values) / len(values) if values else None

    ref_summary = {}
    for name in ("action", "safe_direction", "stop_now", "risk", "answer_exact"):
        correct = reference_scores[f"{name}_correct"]
        total = reference_scores[f"{name}_total"]
        ref_summary[f"{name}_accuracy"] = correct / total if total else None
        ref_summary[f"{name}_total"] = total
    ref_summary["mean_token_f1"] = mean(reference_scores["token_f1"])
    ref_summary["mean_rouge_l"] = mean(reference_scores["rouge_l"])
    ref_summary["mean_bleu4"] = mean(reference_scores["bleu4"])

    total = len(records)
    parsed = sum(isinstance(payload, dict) for payload in parsed_payloads)
    return {
        "path": str(path),
        "total_frames": total,
        "error_frames": errors,
        "json_parse_success_frames": parsed,
        "json_parse_success_rate": parsed / total if total else 0.0,
        "avg_latency_s": mean(latencies),
        "median_latency_s": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "stop_now_true_frames": stop_true,
        "recommended_action_counts": dict(actions),
        "safe_direction_counts": dict(safe_directions),
        "pedestrian_risk_counts": dict(risks),
        "invalid_recommended_action_counts": dict(invalid_actions),
        "invalid_safe_direction_counts": dict(invalid_directions),
        "mean_uncertainty": mean(uncertainty_values),
        "redundancy": {
            "adjacent_exact_repeat_count": exact_repeat_count,
            "adjacent_exact_repeat_rate": exact_repeat_count / max(total - 1, 1) if total > 1 else 0.0,
            "mean_adjacent_text_similarity": mean(adjacent_similarities),
        },
        "reference_metrics": ref_summary,
    }


def main():
    args = parse_args()
    refs = load_references(args.references)
    summaries = [evaluate_file(path, refs) for path in args.results]
    output = {"results": summaries}
    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
