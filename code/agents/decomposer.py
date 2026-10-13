import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def strip_markdown_fences(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    t = re.sub(r"^```(?:json|JSON)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def extract_outermost_json(text: str):
    if not text:
        return None
    stack = []
    start_index = None
    try:
        for i, char in enumerate(text):
            if char == "{":
                if not stack:
                    start_index = i
                stack.append(char)
            elif char == "}":
                if not stack:
                    continue
                stack.pop()
                if not stack and start_index is not None:
                    return text[start_index : i + 1]
    except Exception as e:
        print(f"[extract_outermost_json] error: {e}")
        return None
    return None


def safe_json_loads(json_string: str):
    if not json_string:
        return None
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        try:
            fixed = _single_to_double_quotes_outside_strings(json_string)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"[safe_json_loads] quote-fix exception: {e}")
        try:
            fixed = _single_to_double_quotes_outside_strings(json_string)
            fixed = _python_kw_to_json(fixed)
            return json.loads(fixed)
        except Exception as e:
            print(f"[safe_json_loads] fallback failed: {e}")
            return None
    except Exception as e:
        print(f"[safe_json_loads] error: {e}")
        return None


def _python_kw_to_json(s: str) -> str:
    out = []
    inside = False
    escaped = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '"' and not escaped:
            inside = not inside
        if not inside:
            if s.startswith("True", i) and (i + 4 == len(s) or not s[i + 4].isalnum()):
                out.append("true")
                i += 4
                continue
            if s.startswith("False", i) and (i + 5 == len(s) or not s[i + 5].isalnum()):
                out.append("false")
                i += 5
                continue
            if s.startswith("None", i) and (i + 4 == len(s) or not s[i + 4].isalnum()):
                out.append("null")
                i += 4
                continue
        out.append(ch)
        escaped = ch == "\\" and not escaped
        i += 1
    return "".join(out)


def _single_to_double_quotes_outside_strings(s: str) -> str:
    out = []
    inside = False
    escaped = False
    for ch in s:
        if ch == '"' and not escaped:
            inside = not inside
        if ch == "'" and not inside:
            out.append('"')
        else:
            out.append(ch)
        escaped = ch == "\\" and not escaped
    return "".join(out)


VALID_TOP_KEYS = {"E", "E1", "E2", "E3", "A", "Q", "F_auxiliary"}


def _normalize_verifiable(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return False


def _normalize_subclaim(obj) -> dict:
    if not isinstance(obj, dict):
        return {"content": "N/A", "verifiable": False, "focus_question": None}
    content = obj.get("content", "N/A")
    if not isinstance(content, str):
        content = str(content)
    verifiable = _normalize_verifiable(obj.get("verifiable", False))
    fq = obj.get("focus_question", None)
    if fq in ("", "null", "None"):
        fq = None
    if content.strip().upper() == "N/A":
        verifiable = False
        fq = None
    return {"content": content, "verifiable": verifiable, "focus_question": fq}


def validate_decomposition(raw: dict) -> dict:
    out = {
        "E_list": [],
        "A": None,
        "Q": None,
        "F_auxiliary": {"content": "N/A", "note": "N/A"},
        "parse_ok": True,
        "parse_error": None,
        "compound": False,
    }

    if not isinstance(raw, dict):
        out["parse_ok"] = False
        out["parse_error"] = "raw is not a dict"
        return out

    if "decomposition" in raw and isinstance(raw["decomposition"], dict):
        raw = raw["decomposition"]

    e_keys_exist = any(k for k in raw if k in ("E", "E1", "E2", "E3"))
    if not e_keys_exist and "E_list" in raw and isinstance(raw["E_list"], list):
        e_list_raw = raw["E_list"]
        for idx, item in enumerate(e_list_raw, start=1):
            if isinstance(item, dict):
                raw[f"E{idx}"] = item

    e_keys = [k for k in raw.keys() if k == "E" or (k.startswith("E") and len(k) == 2 and k[1].isdigit())]
    if not e_keys:
        out["parse_ok"] = False
        out["parse_error"] = "no E dimension found (and no convertible E_list)"
        return out

    e_keys_sorted = sorted(e_keys, key=lambda k: (0 if k == "E" else int(k[1:])))
    for k in e_keys_sorted:
        norm = _normalize_subclaim(raw.get(k))
        norm["key"] = k
        out["E_list"].append(norm)
    out["compound"] = len(out["E_list"]) > 1

    out["A"] = _normalize_subclaim(raw.get("A", {"content": "N/A", "verifiable": False, "focus_question": None}))
    out["Q"] = _normalize_subclaim(raw.get("Q", {"content": "N/A", "verifiable": False, "focus_question": None}))

    fa = raw.get("F_auxiliary", {})
    if isinstance(fa, dict):
        out["F_auxiliary"] = {
            "content": str(fa.get("content", "N/A")),
            "note": str(fa.get("note", "N/A")),
        }

    return out


def setup_decomposer(model_name: str, dataset: str):
    from autogen import AssistantAgent, UserProxyAgent
    from PromptH import PromptHandler
    from config import ModelConfig

    handler = PromptHandler()
    model_config = ModelConfig()
    llm_config = model_config.get_config(model_name)

    if dataset in ["RAWFC", "LIAR-RAW"]:
        sys_msg = handler.handle_prompt("Decomposer_en")
        agent_name = "Decomposer"
    else:
        sys_msg = handler.handle_prompt("Decomposer_ch")
        agent_name = "分解器"
    agent_name = agent_name.replace(" ", "_")

    decomposer = AssistantAgent(
        name=agent_name,
        system_message=sys_msg,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content") or ""),
    )
    user_proxy = UserProxyAgent(
        "user_proxy", code_execution_config=False, human_input_mode="NEVER"
    )
    return decomposer, user_proxy


def decompose_claim(claim: str, decomposer, user_proxy, dataset: str = "RAWFC", *,
                    logger=None, event_id: str = "") -> dict:
    if dataset in ["RAWFC", "LIAR-RAW"]:
        user_msg = (
            "Decompose the following claim into E/A/Q sub-claims and F_auxiliary.\n"
            "Output strict JSON only, no explanation, no markdown fences.\n\n"
            f"Claim: {claim}\n"
        )
    else:
        user_msg = (
            "请将以下声明分解为 E/A/Q 子声明和 F_auxiliary。\n"
            "只输出严格 JSON，不要任何解释或 markdown 代码块。\n\n"
            f"声明: {claim}\n"
        )

    start_time = time.perf_counter()
    res = user_proxy.initiate_chat(
        recipient=decomposer,
        clear_history=True,
        message=user_msg,
        cache=None,
        summary_method="last_msg",
        max_turns=1,
    )
    end_time = time.perf_counter()
    raw_text = res.summary or ""

    stripped = strip_markdown_fences(raw_text)
    json_block = extract_outermost_json(stripped)
    if json_block is None:
        json_block = extract_outermost_json(raw_text)

    parsed = safe_json_loads(json_block) if json_block else None
    canonical = validate_decomposition(parsed) if parsed else {
        "E_list": [],
        "A": _normalize_subclaim({}),
        "Q": _normalize_subclaim({}),
        "F_auxiliary": {"content": "N/A", "note": "N/A"},
        "parse_ok": False,
        "parse_error": "could not parse JSON from LLM response",
        "compound": False,
    }

    canonical["claim"] = claim
    canonical["raw_response"] = raw_text
    if logger is not None:
        logger.log_llm_call(
            result=res,
            input_text=user_msg,
            output_text=raw_text,
            context={"event_id": event_id, "stage": "decomposition", "agent": "Decomposer"},
            start_time=start_time,
            end_time=end_time,
            parse_ok=canonical.get("parse_ok"),
        )
    return canonical


def _load_dataset(data_dir: Path, dataset: str, task: str) -> list:
    target = data_dir / dataset / task
    if not target.exists():
        alt = data_dir / task
        if alt.exists():
            target = alt
        else:
            raise FileNotFoundError(f"Could not find {target} or {alt}")

    items = []
    for fname in sorted(os.listdir(target)):
        if fname.endswith(".json"):
            with open(target / fname, "r", encoding="utf-8") as f:
                items.append(json.load(f))
    return items


def _save_result(result: dict, output_file: Path) -> None:
    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="DACV Stage 1: Dimensional Decomposition")
    parser.add_argument("--model_name", type=str, default="deepseek-flash")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data directory (e.g. /path/to/RAWFC_parent or /path/to/RAWFC)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["RAWFC", "LIAR-RAW"],
                        help="Dataset name")
    parser.add_argument("--task", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "results" / "decomposition"),
                        help="Results root; files saved to <output_dir>/<dataset>/<task>/")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Max number of claims to process (-1 = all)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip items already decomposed (idempotent resume)")
    parser.add_argument("--run_dir", type=str, default="",
                        help="Run audit directory for configuration and API logs")
    parser.add_argument("--run_id", type=str, default="",
                        help="Stable identifier written to audit records")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.dataset / args.task
    os.makedirs(output_dir, exist_ok=True)

    logger = None
    if args.run_dir:
        from run_logger import RunLogger
        logger = RunLogger(args.run_dir, run_id=args.run_id or None)
        logger.write_preflight({
            "purpose": "Stage 1 decomposition",
            "dataset": args.dataset,
            "split": args.task,
            "sample_limit": args.limit,
            "model_id_requested": os.environ.get("DACV_MODEL_ID", "deepseek-flash"),
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "temperature": os.environ.get("DACV_TEMPERATURE", "0.1"),
            "thinking": "disabled",
            "api_seed": os.environ.get("DACV_API_SEED"),
            "cache_seed": 42,
            "max_tokens": os.environ.get("DACV_MAX_TOKENS"),
            "stage": "decomposition",
        })

    items = _load_dataset(data_dir, args.dataset, args.task)
    if args.limit > 0:
        items = items[: args.limit]

    decomposer, user_proxy = setup_decomposer(args.model_name, args.dataset)

    n_ok, n_fail, n_compound = 0, 0, 0
    for idx, item in enumerate(items):
        event_id = str(item.get("event_id", f"idx_{idx}")).replace(".json", "")
        output_file = output_dir / f"{event_id}.json"
        if args.skip_existing and output_file.exists():
            print(f"[{idx+1}/{len(items)}] {event_id} exists, skipping.")
            continue

        claim = item["claim"]
        try:
            result = decompose_claim(
                claim, decomposer, user_proxy, dataset=args.dataset,
                logger=logger, event_id=event_id,
            )
        except Exception as e:
            print(f"[{idx+1}/{len(items)}] {event_id} ERROR: {e}")
            result = {
                "claim": claim,
                "E_list": [],
                "A": _normalize_subclaim({}),
                "Q": _normalize_subclaim({}),
                "F_auxiliary": {"content": "N/A", "note": "N/A"},
                "parse_ok": False,
                "parse_error": f"exception: {type(e).__name__}: {e}",
                "compound": False,
                "raw_response": "",
            }

        record = {
            "event_id": event_id,
            "claim": claim,
            "label": item.get("label"),
            "decomposition": result,
        }
        _save_result(record, output_file)
        if logger is not None:
            logger.log_sample({
                "event_id": event_id,
                "stage": "decomposition",
                "output_path": str(output_file),
                "parse_ok": result.get("parse_ok"),
                "n_event_subclaims": len(result.get("E_list", [])),
                "has_attribution": bool(result.get("A", {}).get("verifiable")),
                "has_quantity": bool(result.get("Q", {}).get("verifiable")),
            })

        if result["parse_ok"]:
            n_ok += 1
            if result.get("compound"):
                n_compound += 1
        else:
            n_fail += 1

        print(
            f"[{idx+1}/{len(items)}] {event_id} "
            f"{'OK' if result['parse_ok'] else 'FAIL'}"
            f"{' (compound)' if result.get('compound') else ''}"
        )

    print("\n=== Decomposition summary ===")
    print(f"total   : {len(items)}")
    print(f"ok      : {n_ok}")
    print(f"fail    : {n_fail}")
    print(f"compound: {n_compound}")
    print(f"output  : {output_dir}")


if __name__ == "__main__":
    main()
