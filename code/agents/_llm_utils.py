import json
import re


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
                    return text[start_index: i + 1]
    except Exception as e:
        print(f"[extract_outermost_json] error: {e}")
        return None
    return None


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


def wrap_llm_config(raw_cfg: dict) -> dict:
    if "config_list" in raw_cfg:
        return raw_cfg
    inner = {k: v for k, v in raw_cfg.items() if k not in ("cache_seed", "temperature")}
    wrapped = {"config_list": [inner]}
    if "cache_seed" in raw_cfg:
        wrapped["cache_seed"] = raw_cfg["cache_seed"]
    if "temperature" in raw_cfg:
        wrapped["temperature"] = raw_cfg["temperature"]
    return wrapped


_wrap_llm_config = wrap_llm_config
