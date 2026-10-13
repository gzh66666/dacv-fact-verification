import json
import os
from pathlib import Path


def build_pool_from_rawfc_item(item: dict, dedup: bool = True) -> list:
    pool = []
    seen = set()

    for report in item.get("reports", []):
        for sent_obj in report.get("tokenized", []):
            if sent_obj.get("is_evidence") == 1:
                s = (sent_obj.get("sent") or "").strip()
                if not s:
                    continue
                if sent_obj.get("is_repeat") == 1:
                    continue
                if dedup:
                    key = s.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                pool.append(s)

    return pool


def save_pool(event_id: str, claim: str, label: str, pool: list, output_file: Path) -> None:
    os.makedirs(output_file.parent, exist_ok=True)
    payload = {
        "event_id": event_id,
        "claim": claim,
        "label": label,
        "evidence_pool": pool,
        "pool_size": len(pool),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_pool_for_dataset(data_dir: Path, output_dir: Path, dataset: str = "RAWFC") -> dict:
    if dataset not in ("RAWFC", "LIAR-RAW"):
        raise NotImplementedError(
            "evidence_pool builder only supports RAWFC and LIAR-RAW directly."
        )

    sizes = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(data_dir / fname, "r", encoding="utf-8") as f:
            item = json.load(f)
        pool = build_pool_from_rawfc_item(item)
        event_id = str(item["event_id"]).replace(".json", "")
        save_pool(
            event_id=event_id,
            claim=item["claim"],
            label=item["label"],
            pool=pool,
            output_file=output_dir / f"{event_id}.json",
        )
        sizes.append(len(pool))

    if sizes:
        summary = {
            "num_items": len(sizes),
            "min": min(sizes),
            "max": max(sizes),
            "mean": round(sum(sizes) / len(sizes), 2),
            "median": sorted(sizes)[len(sizes) // 2],
        }
    else:
        summary = {"num_items": 0}
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build evidence pools from RAWFC is_evidence annotations.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="e.g. /path/to/RAWFC/test")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="e.g. results/evidence_pool/RAWFC/test")
    parser.add_argument("--dataset", type=str, default="RAWFC", choices=["RAWFC", "LIAR-RAW"])
    args = parser.parse_args()

    summary = build_pool_for_dataset(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        dataset=args.dataset,
    )
    print("Evidence pool summary:", json.dumps(summary, indent=2))
