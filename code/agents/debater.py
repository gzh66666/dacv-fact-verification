import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from agents._llm_utils import (  # noqa: E402
    strip_markdown_fences,
    extract_outermost_json,
    safe_json_loads,
    wrap_llm_config,
)


def _normalize_argument(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return {
            "argument": "",
            "evidence_used": [],
            "concessions": "none",
            "needs_external_search": False,
            "search_query": "",
            "parse_ok": False,
            "parse_error": "raw is not a dict",
        }

    out = {
        "argument": str(raw.get("argument", "")).strip(),
        "evidence_used": [],
        "concessions": str(raw.get("concessions", "none")).strip() or "none",
        "needs_external_search": bool(raw.get("needs_external_search", False)),
        "search_query": str(raw.get("search_query", "")).strip(),
        "parse_ok": True,
        "parse_error": None,
    }

    ev = raw.get("evidence_used", [])
    if isinstance(ev, list):
        out["evidence_used"] = [str(s).strip() for s in ev if str(s).strip()]
    elif isinstance(ev, str) and ev.strip():
        out["evidence_used"] = [ev.strip()]

    if out["needs_external_search"] and not out["search_query"]:
        out["needs_external_search"] = False

    return out


def setup_debater(model_name: str, dataset: str, side: str):
    if side not in ("Pro", "Con"):
        raise ValueError(f"side must be 'Pro' or 'Con', got {side!r}")

    from autogen import AssistantAgent, UserProxyAgent
    from PromptH import PromptHandler
    from config import ModelConfig

    handler = PromptHandler()
    model_config = ModelConfig()
    raw_cfg = model_config.get_config(model_name)
    llm_config = wrap_llm_config(raw_cfg)

    if dataset in ["RAWFC", "LIAR-RAW"]:
        prompt_key = f"{side}_Debater_en"
    else:
        prompt_key = f"{side}_Debater_ch"

    sys_msg = handler.handle_prompt(prompt_key)
    if sys_msg.startswith("Prompt file ") or sys_msg == "Invalid prompt name":
        raise RuntimeError(
            f"PromptHandler could not load '{prompt_key}'. "
            f"Make sure {prompt_key}.txt exists in the prompts/ folder "
            f"and PromptH.py registers it (or uses the env-var fallback)."
        )

    agent_name = f"{side}_Debater"
    debater = AssistantAgent(
        name=agent_name,
        system_message=sys_msg,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content") or ""),
    )
    user_proxy = UserProxyAgent(
        f"user_proxy_{side.lower()}", code_execution_config=False, human_input_mode="NEVER"
    )
    return debater, user_proxy


def run_argument(
    debater,
    user_proxy,
    *,
    side: str,
    dataset: str,
    original_claim: str,
    sub_claim_text: str,
    sub_claim_dimension: str,
    focus_question: str,
    evidence_sentences: List[str],
    opponent_round1: Optional[dict] = None,
    logger=None,
    call_context: Optional[dict] = None,
) -> dict:
    is_en = dataset in ["RAWFC", "LIAR-RAW"]

    ev_block = "\n".join(f"- {s}" for s in evidence_sentences) if evidence_sentences else (
        "(empty)" if is_en else "（空）"
    )

    if is_en:
        msg = (
            f"ORIGINAL CLAIM: {original_claim}\n\n"
            f"SUB-CLAIM ({sub_claim_dimension}): {sub_claim_text}\n\n"
            f"FOCUS QUESTION: {focus_question}\n\n"
            f"EVIDENCE POOL:\n{ev_block}\n"
        )
        if opponent_round1:
            opp_arg = opponent_round1.get("argument", "")
            opp_ev = opponent_round1.get("evidence_used", []) or []
            opp_block = (
                f"\n\nOPPONENT ROUND 1 ARGUMENT:\n{opp_arg}\n"
                f"OPPONENT EVIDENCE:\n" + "\n".join(f"- {s}" for s in opp_ev)
            )
            msg += opp_block + "\n\nNow produce your Round 2 argument, rebutting where you can."
        else:
            msg += "\nProduce your Round 1 argument."
        msg += "\n\nOutput strict JSON only."
    else:
        msg = (
            f"原始声明：{original_claim}\n\n"
            f"子声明（{sub_claim_dimension}）：{sub_claim_text}\n\n"
            f"聚焦问题：{focus_question}\n\n"
            f"证据池：\n{ev_block}\n"
        )
        if opponent_round1:
            opp_arg = opponent_round1.get("argument", "")
            opp_ev = opponent_round1.get("evidence_used", []) or []
            opp_block = (
                f"\n\n对手第一轮论证：\n{opp_arg}\n"
                f"对手引用的证据：\n" + "\n".join(f"- {s}" for s in opp_ev)
            )
            msg += opp_block + "\n\n现在产出你的第二轮论证，可对对手论点进行反驳。"
        else:
            msg += "\n请产出你的第一轮论证。"
        msg += "\n\n只输出严格 JSON。"

    start_time = time.perf_counter()
    res = user_proxy.initiate_chat(
        recipient=debater,
        clear_history=True,
        message=msg,
        cache=None,
        summary_method="last_msg",
        max_turns=1,
    )
    end_time = time.perf_counter()
    raw_text = res.summary or ""

    stripped = strip_markdown_fences(raw_text)
    json_block = extract_outermost_json(stripped) or extract_outermost_json(raw_text)
    parsed = safe_json_loads(json_block) if json_block else None

    canonical = _normalize_argument(parsed if parsed else {})
    canonical["side"] = side
    canonical["dimension"] = sub_claim_dimension
    canonical["raw_response"] = raw_text
    if logger is not None:
        logger.log_llm_call(
            result=res,
            input_text=msg,
            output_text=raw_text,
            context=call_context or {"stage": "debate", "agent": side},
            start_time=start_time,
            end_time=end_time,
            parse_ok=canonical.get("parse_ok"),
        )
    return canonical
