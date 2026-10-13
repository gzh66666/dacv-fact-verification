from typing import Dict, Optional
import re


VALID_LABELS = ("true", "half", "false")


STRONG_FRAMING_TERMS = {
    "proved", "definitely", "absolutely", "certainly", "literally",
    "always", "never", "inevitably", "completely",
    "ruined", "wiped out", "catastrophic", "devastating",
    "shockingly", "outrageous", "creepy", "flesh-eating",
}

NO_FRAMING_SENTINELS = {
    "no loaded wording", "n/a", "none", "no framing", "neutral",
}


def assess_framing_strength(f_auxiliary: dict) -> str:
    if not f_auxiliary or not isinstance(f_auxiliary, dict):
        return "none"

    content = str(f_auxiliary.get("content", "")).strip().lower()
    if not content or content in NO_FRAMING_SENTINELS:
        return "none"

    matched = [
        term for term in STRONG_FRAMING_TERMS
        if re.search(
            r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)",
            content,
            re.IGNORECASE,
        )
    ]
    if matched:
        return "strong"

    return "weak"


LITERAL_MISMATCH_SIGNALS = [
    r"\bproposed rule\b",
    r"\bproposed regulation\b",
    r"\bnever (?:enacted|took effect|finalized|implemented)\b",
    r"\bwithdrew (?:a |the )?proposed\b",
    r"\bwithdrawal of (?:a |the )?proposed\b",
    r"\bwas (?:a |the )?proposal\b",
    r"\b(?:never|hadn't) (?:gone|come) into (?:effect|force)\b",

    r"\bby (?:her|his) aide\b",
    r"\bby an aide\b",
    r"\bdone by (?:her|his) aide\b",
    r"\bperformed by (?:her|his) aide\b",
    r"\bthe aide,?\s+(?:who|named)\b",
    r"\b(?:her|his) staffer (?:did|performed)\b",

    r"\bany brand of\b",
    r"\bregardless of (?:the )?brand\b",
    r"\b(?:all|any) toilet paper\b",
    r"\bnot (?:specific|unique) to (?:Costco|the brand|this)\b",
    r"\bgeneric (?:to all|across all)\b",

    r"\bnot (?:a |an )?newspaper\b",
    r"\bwas (?:a |the )?flyer\b",
    r"\bwas (?:a |the )?leaflet\b",
    r"\bnot an (?:advertisement|ad)\b",

    r"\bfor (?:just |only )?(?:one|a single) day\b",
    r"\bonly for (?:a few hours|one day|a day)\b",
    r"\b(?:was|is|were) (?:only )?temporary\b",
    r"\btemporarily (?:removed|displaced|swapped|replaced)\b",
    r"\bonly lasted (?:a day|one day|a few hours)\b",
    r"\b(?:was|were) reinstated (?:the next day|shortly|hours later)\b",

    r"\bin some states\b(?!\s+of\b)",
    r"\bin (?:two|three|four|five|six|seven|eight|nine|ten|several|a few) states\b",
    r"\b(?:only |just )?a small percentage of\b",
    r"\bminority of (?:users|people|employees|workers|customers)\b",
    r"\b(?:only |just )?(?:a small|a tiny) fraction\b",
    r"\b(?:isolated|individual) cases (?:only|in)\b",

    r"\binitially (?:offered|refunded|paid|denied)\b",
    r"\bfirst offer(?:ed)? (?:was|only)\b",
    r"\bat first (?:offered|refunded|denied|claimed)\b",
    r"\blater (?:resolved|settled|paid in full|reversed)\b",
    r"\beventually (?:paid|refunded|compensated)\b",
    r"\b(?:a |the )?new check (?:was )?(?:issued|sent)\b",
    r"\bsubsequently (?:paid|resolved|reversed|compensated)\b",
    r"\bafter (?:the )?(?:article|story|news|publication) (?:was )?published\b",

    r"\bno formal (?:ban|rule|policy|action)\b",
    r"\bno official (?:ban|action|policy)\b",
    r"\bno record of (?:a |an )?(?:ban|formal)\b",
    r"\b(?:there was )?never a formal\b",

    r"\bfiled for bankruptcy.{0,80}(?:not directly|but.{0,30}cause)\b",
    r"\bindirect (?:involvement|cause|relation|role)\b",
    r"\bdistanced (?:himself|themselves) from\b",

    r"\b(?:only )?(?:a |the )?small minority\b",
    r"\bvery few (?:users|people|cases|instances|tweets)\b",
    r"\b(?:was )?an isolated (?:incident|case|instance)\b",
    r"\b(?:just |only )?a handful of (?:users|cases|complaints|tweets)\b",
]


_LITERAL_MISMATCH_RE = re.compile(
    "|".join(LITERAL_MISMATCH_SIGNALS),
    re.IGNORECASE,
)


def detect_literal_mismatch(reasoning_text: str) -> bool:
    if not reasoning_text:
        return False
    return bool(_LITERAL_MISMATCH_RE.search(reasoning_text))


def dacv_label_v4_with_rule(
    E: str,
    A: str,
    Q: str,
    framing_strength: str = "none",
    literal_mismatch: bool = False,
    E_conf: str = "high",
) -> tuple[str, str]:
    if E == "F":
        if literal_mismatch:
            return "half", "R1a"
        if A == "T" or Q == "T":
            return "half", "R2"

        if framing_strength == "strong":
            return "half", "R1b"


        return "false", "R1c"

    if E == "T" and A in ("T", "N/A") and Q in ("T", "N/A"):
        if framing_strength != "strong" and E_conf == "high":
            return "true", "R3"
        if framing_strength == "strong" or E_conf != "high":
            return "half", "R3.5"

    if E == "T" and A == "F":
        return "half", "R4"

    if E == "T" and Q == "F":
        return "half", "R5"

    active = [v for v in (E, A, Q) if v != "N/A"]
    n_u = sum(1 for v in active if v == "U")
    n_t = sum(1 for v in active if v == "T")

    if n_u == len(active) and n_u > 0:
        return "half", "R6-default"

    if n_u >= 1 and n_t >= 1 and E == "T" and (A == "T" or Q == "T"):
        if framing_strength != "strong":
            return "true", "R6-true-exception"
        return "half", "R6-default"

    if n_u >= 1:
        return "half", "R6-default"

    return "half", "R6-default"


def dacv_label_v4(
    E: str,
    A: str,
    Q: str,
    framing_strength: str = "none",
    literal_mismatch: bool = False,
    E_conf: str = "high",
) -> str:
    label, _ = dacv_label_v4_with_rule(
        E, A, Q, framing_strength, literal_mismatch, E_conf
    )
    return label


def dacv_label_v3(E, A, Q, framing_strength="none"):
    return dacv_label_v4(E, A, Q, framing_strength=framing_strength,
                         literal_mismatch=False, E_conf="high")


def dacv_label_v2(E, A, Q):
    return dacv_label_v4(E, A, Q, framing_strength="none",
                         literal_mismatch=False, E_conf="high")


def extract_dimension_verdicts(stage2_record: dict) -> Dict[str, dict]:
    out = {}
    vs = stage2_record.get("verdicts_summary", {})
    debates = stage2_record.get("debates", [])

    e_verdicts = vs.get("E_verdicts", [])
    if e_verdicts:
        all_v = [e["verdict"] for e in e_verdicts]
        all_c = [e.get("confidence", "low") for e in e_verdicts]
        if "F" in all_v:
            agg_v = "F"
            agg_c = min((c for v, c in zip(all_v, all_c) if v == "F"),
                        key=lambda c: {"high": 2, "medium": 1, "low": 0}.get(c, 0))
        elif "U" in all_v:
            agg_v = "U"
            agg_c = "low"
        elif all(v == "T" for v in all_v):
            agg_v = "T"
            agg_c = min(all_c, key=lambda c: {"high": 2, "medium": 1, "low": 0}.get(c, 0))
        else:
            agg_v = "N/A"
            agg_c = "low"
        e_debates = [d for d in debates if d.get("dimension") == "E"]
        combined_reasoning = " | ".join(
            f"[{d.get('key','E')}] {d.get('verdict',{}).get('reasoning','')}"
            for d in e_debates
        )
        combined_focus = " | ".join(
            f"[{d.get('key','E')}] {d.get('focus_question','')}"
            for d in e_debates
        )
        out["E"] = {
            "verdict": agg_v,
            "confidence": agg_c,
            "reasoning": combined_reasoning,
            "focus_question": combined_focus,
            "n_sub_events": len(e_verdicts),
            "sub_verdicts": [(e["verdict"], e.get("confidence","")) for e in e_verdicts],
        }
    else:
        out["E"] = {"verdict": "N/A", "confidence": "low", "reasoning": "", "focus_question": "",
                    "n_sub_events": 0, "sub_verdicts": []}

    for letter in ("A", "Q"):
        v = vs.get(f"{letter}_verdict")
        d = next((db for db in debates if db.get("dimension") == letter), None)
        if v and d:
            out[letter] = {
                "verdict": v["verdict"],
                "confidence": v["confidence"],
                "reasoning": d.get("verdict", {}).get("reasoning", ""),
                "focus_question": d.get("focus_question", ""),
            }
        else:
            out[letter] = {"verdict": "N/A", "confidence": "low", "reasoning": "", "focus_question": ""}

    return out


def derive_label_with_trigger(
    stage2_record: dict,
    f_auxiliary: Optional[dict] = None,
) -> dict:
    dims = extract_dimension_verdicts(stage2_record)
    framing = assess_framing_strength(f_auxiliary or {})

    combined_reasoning = (
        dims["E"].get("reasoning", "") + " " + dims["A"].get("reasoning", "")
    )
    lit_mismatch = detect_literal_mismatch(combined_reasoning)

    rule_label, rule_id = dacv_label_v4_with_rule(
        dims["E"]["verdict"],
        dims["A"]["verdict"],
        dims["Q"]["verdict"],
        framing_strength=framing,
        literal_mismatch=lit_mismatch,
        E_conf=dims["E"]["confidence"],
    )

    return {
        "event_id": stage2_record.get("event_id"),
        "claim": stage2_record.get("claim"),
        "gold_label": stage2_record.get("label"),
        "dim_verdicts": dims,
        "framing_strength": framing,
        "framing_content": (f_auxiliary or {}).get("content", "N/A"),
        "literal_mismatch": lit_mismatch,
        "rule_label": rule_label,
        "rule_id": rule_id,
    }


