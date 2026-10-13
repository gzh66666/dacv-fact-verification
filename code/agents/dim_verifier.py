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


VALID_VERDICTS = {"T", "F", "U"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def _normalize_verdict(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return {
            "verdict": "U",
            "confidence": "low",
            "reasoning": "verifier output was not a dict",
            "decisive_evidence": [],
            "parse_ok": False,
            "parse_error": "raw is not a dict",
        }

    v = str(raw.get("verdict", "U")).strip().upper()
    if v not in VALID_VERDICTS:
        if v in ("TRUE", "T."):
            v = "T"
        elif v in ("FALSE", "F."):
            v = "F"
        elif v in ("UNVERIFIABLE", "UNCERTAIN", "U."):
            v = "U"
        else:
            v = "U"

    conf = str(raw.get("confidence", "low")).strip().lower()
    if conf not in VALID_CONFIDENCE:
        conf = "low"

    reasoning = str(raw.get("reasoning", "")).strip()

    de = raw.get("decisive_evidence", [])
    if isinstance(de, list):
        decisive = [str(s).strip() for s in de if str(s).strip()]
    elif isinstance(de, str) and de.strip():
        decisive = [de.strip()]
    else:
        decisive = []

    return {
        "verdict": v,
        "confidence": conf,
        "reasoning": reasoning,
        "decisive_evidence": decisive,
        "parse_ok": True,
        "parse_error": None,
    }


def setup_verifier(model_name: str, dataset: str):
    from autogen import AssistantAgent, UserProxyAgent
    from PromptH import PromptHandler
    from config import ModelConfig

    handler = PromptHandler()
    model_config = ModelConfig()
    raw_cfg = model_config.get_config(model_name)
    llm_config = wrap_llm_config(raw_cfg)

    prompt_key = "Dim_Verifier_en" if dataset in ["RAWFC", "LIAR-RAW"] else "Dim_Verifier_ch"
    sys_msg = handler.handle_prompt(prompt_key)
    if sys_msg.startswith("Prompt file ") or sys_msg == "Invalid prompt name":
        raise RuntimeError(
            f"PromptHandler could not load '{prompt_key}'. "
            f"Make sure {prompt_key}.txt is in prompts/."
        )

    verifier = AssistantAgent(
        name="Dim_Verifier",
        system_message=sys_msg,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content") or ""),
    )
    user_proxy = UserProxyAgent(
        "user_proxy_verifier", code_execution_config=False, human_input_mode="NEVER"
    )
    return verifier, user_proxy


def run_verifier(
    verifier,
    user_proxy,
    *,
    dataset: str,
    sub_claim_text: str,
    sub_claim_dimension: str,
    focus_question: str,
    evidence_sentences: List[str],
    pro_arguments: List[dict],
    con_arguments: List[dict],
    logger=None,
    call_context: Optional[dict] = None,
) -> dict:
    is_en = dataset in ["RAWFC", "LIAR-RAW"]

    ev_block = "\n".join(f"- {s}" for s in evidence_sentences) if evidence_sentences else (
        "(empty)" if is_en else "（空）"
    )

    def _fmt_round(side: str, idx: int, arg: dict, lang_en: bool) -> str:
        prefix = f"{side} Round {idx + 1}" if lang_en else f"{side} 第{idx + 1}轮"
        ev_used = arg.get("evidence_used") or []
        ev_str = "\n".join(f"  - {s}" for s in ev_used) if ev_used else (
            "  (none)" if lang_en else "  （无）"
        )
        body_label = "Argument" if lang_en else "论证"
        ev_label = "Evidence cited" if lang_en else "引用证据"
        conc_label = "Concessions" if lang_en else "承认部分"
        return (
            f"\n=== {prefix} ===\n"
            f"{body_label}: {arg.get('argument', '')}\n"
            f"{ev_label}:\n{ev_str}\n"
            f"{conc_label}: {arg.get('concessions', 'none')}\n"
        )

    pro_block = "".join(_fmt_round("PRO", i, a, is_en) for i, a in enumerate(pro_arguments))
    con_block = "".join(_fmt_round("CON", i, a, is_en) for i, a in enumerate(con_arguments))

    if is_en:
        msg = (
            f"SUB-CLAIM ({sub_claim_dimension}): {sub_claim_text}\n\n"
            f"FOCUS QUESTION: {focus_question}\n\n"
            f"EVIDENCE POOL:\n{ev_block}\n"
            f"{pro_block}{con_block}\n"
            "Now issue your verdict. Output strict JSON only."
        )
    else:
        msg = (
            f"子声明（{sub_claim_dimension}）：{sub_claim_text}\n\n"
            f"聚焦问题：{focus_question}\n\n"
            f"证据池：\n{ev_block}\n"
            f"{pro_block}{con_block}\n"
            "现在做出裁决。只输出严格 JSON。"
        )

    start_time = time.perf_counter()
    res = user_proxy.initiate_chat(
        recipient=verifier,
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

    canonical = _normalize_verdict(parsed if parsed else {})
    canonical["dimension"] = sub_claim_dimension
    canonical["raw_response"] = raw_text
    if logger is not None:
        logger.log_llm_call(
            result=res,
            input_text=msg,
            output_text=raw_text,
            context=call_context or {"stage": "verifier", "agent": "Dim_Verifier"},
            start_time=start_time,
            end_time=end_time,
            parse_ok=canonical.get("parse_ok"),
        )
    return canonical
