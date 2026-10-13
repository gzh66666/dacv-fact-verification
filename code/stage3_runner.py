import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from rules.label_rules import derive_label_with_trigger  # noqa: E402


# Stage 2 trace contract: every record in <stage2_dir>/<dataset>/<task>/ carries
# "debates", "verdicts_summary" and "label". Stage 3 reads them together with
# the optional Stage 1 "F_auxiliary" framing field and writes one record per
# claim with "final_label", "rule_id", "framing_strength" and "literal_mismatch".
# rule_id values are emitted by rules/label_rules.py: R1a, R1b, R1c, R2, R3,
# R3.5, R4, R5, R6-true-exception (paper R9) and R6-default (paper R10).


def _load_stage2(stage2_dir: Path, dataset: str, task: str) -> List[dict]:
    target = stage2_dir / dataset / task
    if not target.exists():
        raise FileNotFoundError(f"No Stage 2 outputs at {target}")
    items = []
    for fn in sorted(os.listdir(target)):
        if fn.endswith(".json"):
            with open(target / fn, "r", encoding="utf-8") as f:
                try:
                    items.append(json.load(f))
                except json.JSONDecodeError:
                    print(f"[warn] skipping malformed json: {fn}")
    return items


def _load_stage1_f_auxiliary(decomp_dir: Path, dataset: str, task: str, event_id: str) -> Optional[dict]:
    p = decomp_dir / dataset / task / f"{event_id}.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return d.get("decomposition", {}).get("F_auxiliary")


def _save_stage3(record: dict, output_file: Path) -> None:
    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def run_stage3_for_item(
    stage2_record: dict,
    *,
    f_auxiliary: Optional[dict] = None,
) -> dict:
    rule = derive_label_with_trigger(stage2_record, f_auxiliary=f_auxiliary)
    return {
        "event_id": stage2_record.get("event_id"),
        "claim": stage2_record.get("claim"),
        "gold_label": stage2_record.get("label"),
        "dim_verdicts": rule["dim_verdicts"],
        "framing_strength": rule["framing_strength"],
        "framing_content": rule["framing_content"],
        "literal_mismatch": rule["literal_mismatch"],
        "rule_label": rule["rule_label"],
        "rule_id": rule["rule_id"],
        "final_label": rule["rule_label"],
    }


def evaluate_records(records: List[dict], pred_field: str = "final_label") -> dict:
    labels = ("true", "half", "false")
    n = len(records)
    n_correct = sum(1 for r in records if r[pred_field] == r["gold_label"])

    cm = {g: Counter() for g in labels}
    for r in records:
        cm[r["gold_label"]][r[pred_field]] += 1

    per_class = {}
    for cls in labels:
        tp = cm[cls][cls]
        fp = sum(cm[g][cls] for g in labels if g != cls)
        fn = sum(cm[cls][p] for p in labels if p != cls)
        support = sum(cm[cls].values())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[cls] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    return {
        "n": n,
        "accuracy": n_correct / n if n else 0.0,
        "per_class": per_class,
        "macro_f1": sum(per_class[c]["f1"] for c in labels) / len(labels),
        "confusion_matrix": {g: dict(cm[g]) for g in labels},
    }


def print_eval(metrics: dict, title: str) -> None:
    print(f"\n========== {title} ==========")
    print(f"n = {metrics['n']}, accuracy = {metrics['accuracy']:.1%}, macro-F1 = {metrics['macro_f1']:.3f}")
    print(f"\n{'class':<8} {'support':<8} {'TP':<5} {'FP':<5} {'FN':<5} {'Prec':<8} {'Rec':<8} {'F1'}")
    print("-" * 60)
    for cls in ("true", "half", "false"):
        d = metrics["per_class"][cls]
        print(
            f"{cls:<8} {d['support']:<8} {d['tp']:<5} {d['fp']:<5} {d['fn']:<5} "
            f"{d['precision']:.1%}    {d['recall']:.1%}    {d['f1']:.3f}"
        )

    print("\nConfusion Matrix (rows=gold, cols=pred):")
    print(f"{'':<8} {'true':<6} {'half':<6} {'false':<6}")
    for g in ("true", "half", "false"):
        row = metrics["confusion_matrix"][g]
        print(f"{g:<8} {row.get('true', 0):<6} {row.get('half', 0):<6} {row.get('false', 0):<6}")


def print_errors(records: List[dict], pred_field: str = "final_label", limit: int = 25) -> None:
    print("\n========== Error Samples ==========")
    errors = [r for r in records if r[pred_field] != r["gold_label"]]
    print(f"({len(errors)} errors out of {len(records)})\n")
    for r in errors[:limit]:
        dims = r["dim_verdicts"]
        e = f"E={dims['E']['verdict']}({dims['E']['confidence'][0]})"
        a = f"A={dims['A']['verdict']}({dims['A']['confidence'][0]})" if dims["A"]["verdict"] != "N/A" else "A=N/A"
        q = f"Q={dims['Q']['verdict']}({dims['Q']['confidence'][0]})" if dims["Q"]["verdict"] != "N/A" else "Q=N/A"
        print(
            f"  {r['event_id']} gold={r['gold_label']:<6} pred={r[pred_field]:<6} | "
            f"rule={r['rule_label']:<6} | {e} {a} {q}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="DACV Stage 3: deterministic compositional label derivation")
    parser.add_argument("--model_name", type=str, default="deepseek-v4-flash",
                        help="Kept for command compatibility; Stage3 rule-only does not call an LLM.")
    parser.add_argument("--dataset", type=str, required=True, choices=["RAWFC", "LIAR-RAW"])
    parser.add_argument("--task", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--stage2_dir", type=str, default=str(ROOT / "results" / "stage2"))
    parser.add_argument("--decomp_dir", type=str, default=str(ROOT / "results" / "decomposition"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "results" / "stage3"))
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--run_dir", type=str, default="",
                        help="Run audit directory for metrics and sample index")
    parser.add_argument("--run_id", type=str, default="",
                        help="Stable identifier written to audit records")
    args = parser.parse_args()

    stage2_dir = Path(args.stage2_dir)
    decomp_dir = Path(args.decomp_dir)
    output_dir = Path(args.output_dir) / args.dataset / args.task
    os.makedirs(output_dir, exist_ok=True)

    logger = None
    if args.run_dir:
        from run_logger import RunLogger
        logger = RunLogger(args.run_dir, run_id=args.run_id or None)
        logger.write_preflight({
            "purpose": "Stage 3 deterministic label composition and evaluation",
            "dataset": args.dataset, "split": args.task,
            "model_id_requested": os.environ.get("DACV_MODEL_ID", "deepseek-v4-flash"),
            "stage": "stage3",
        })

    items = _load_stage2(stage2_dir, args.dataset, args.task)
    if args.limit > 0:
        items = items[: args.limit]
    print(f"Loaded {len(items)} Stage 2 records")

    records = []
    t_start = time.time()
    for idx, item in enumerate(items):
        eid = str(item.get("event_id", f"idx_{idx}"))
        out_file = output_dir / f"{eid}.json"
        if args.skip_existing and out_file.exists():
            with open(out_file, "r", encoding="utf-8") as f:
                record = json.load(f)
            records.append(record)
            continue

        try:
            f_aux = _load_stage1_f_auxiliary(decomp_dir, args.dataset, args.task, eid)
            record = run_stage3_for_item(item, f_auxiliary=f_aux)
            _save_stage3(record, out_file)
            records.append(record)
            if logger is not None:
                logger.log_sample({
                    "event_id": eid, "stage": "stage3",
                    "output_path": str(out_file),
                    "gold_label": record.get("gold_label"),
                    "final_label": record.get("final_label"),
                    "rule_id": record.get("rule_id"),
                    "framing_strength": record.get("framing_strength"),
                    "literal_mismatch": record.get("literal_mismatch"),
                    "dim_verdicts": record.get("dim_verdicts"),
                })
            mark = "PASS" if record["final_label"] == record["gold_label"] else "FAIL"
            print(
                f"[{idx + 1}/{len(items)}] {eid} gold={record['gold_label']:<6} "
                f"rule={record['rule_label']:<6} final={record['final_label']:<6} {mark}"
            )
        except Exception as exc:
            print(f"[{idx + 1}/{len(items)}] {eid} ERROR: {type(exc).__name__}: {exc}")

    elapsed = time.time() - t_start
    print(f"\nProcessed {len(records)} records in {elapsed:.1f}s")

    metrics = evaluate_records(records, pred_field="final_label")
    if logger is not None:
        logger.write_metrics({
            "dataset": args.dataset, "split": args.task,
            "model_id_requested": os.environ.get("DACV_MODEL_ID", "deepseek-v4-flash"),
            "metrics": metrics, "n_records": len(records),
        })
    print_eval(metrics, title="DACV Compositional Rule")
    print_errors(records, pred_field="final_label")
    print(
        f"\n[SUMMARY] dataset={args.dataset} n={metrics['n']} "
        f"acc={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}"
    )


if __name__ == "__main__":
    main()
