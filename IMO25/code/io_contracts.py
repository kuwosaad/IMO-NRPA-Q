from __future__ import annotations

"""
io_contracts.py

Lightweight JSON parsing, single-pass repair, and role-specific validators
for Strategist, Worker, and Verifier outputs. Designed to be non-destructive
and easily pluggable into the existing pipeline.

Policies implemented:
- Strict parse with a single repair attempt (trim to first/last JSON brace or bracket,
  strip code fences, normalize quotes) before falling back.
- Validation of role contracts with minimal assumptions; accepts legacy formats
  and normalizes into the new schema when possible.
- One-shot re-ask guidance text builders for precise corrective prompts.
"""

from typing import Any, Dict, List, Tuple, Optional
import json
import re


def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    # Remove leading/trailing markdown code fences if present
    return re.sub(r"^\s*```(json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)


def _extract_json_slice(text: str) -> Optional[str]:
    """Return substring spanning the first '{' to last '}' or first '[' to last ']'."""
    if not text:
        return None
    s = text
    lb, rb = s.find("{"), s.rfind("}")
    if lb != -1 and rb != -1 and rb > lb:
        return s[lb : rb + 1]
    lb, rb = s.find("["), s.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        return s[lb : rb + 1]
    return None


def parse_json_strict_or_repair(text: str) -> Tuple[Optional[Any], str]:
    """
    Try strict json.loads; if it fails, do a single-pass repair by slicing to the
    outermost braces/brackets and stripping code fences. Returns (obj, source)
    where source in {"strict_json", "repaired_json", "parse_failed"}.
    """
    text = text or ""
    try:
        return json.loads(text), "strict_json"
    except Exception:
        pass
    cleaned = _strip_code_fences(text)
    slice_ = _extract_json_slice(cleaned)
    if slice_:
        try:
            return json.loads(slice_), "repaired_json"
        except Exception:
            return None, "parse_failed"
    return None, "parse_failed"


# --------------------------- Strategist ---------------------------
def normalize_strategist_refinements(payload: Any) -> Tuple[List[str], bool, List[str]]:
    """
    Accepts multiple legacy formats and normalizes to (refinements, truncated, errors).
    Supported inputs:
    - {"refinements": [...], "truncated": bool}
    - {"strategies": [...]}  (legacy)
    - ["..."]                (legacy array)
    """
    errors: List[str] = []
    truncated = False
    refinements: List[str] = []

    try:
        if isinstance(payload, dict):
            if "refinements" in payload and isinstance(payload["refinements"], list):
                refinements = [str(x).strip() for x in payload["refinements"] if str(x).strip()]
            elif "strategies" in payload and isinstance(payload["strategies"], list):
                refinements = [str(x).strip() for x in payload["strategies"] if str(x).strip()]
            truncated = bool(payload.get("truncated", False))
        elif isinstance(payload, list):
            refinements = [str(x).strip() for x in payload if str(x).strip()]
        else:
            errors.append("Unsupported strategist payload type")
    except Exception as e:
        errors.append(f"error: {e}")

    # Enforce constraints: 3–5 items, reasonable lengths
    if not refinements:
        errors.append("missing refinements")
    refinements = [r for r in refinements if 1 <= len(r.split()) <= 60 and 1 <= len(r) <= 200]
    if len(refinements) < 3:
        errors.append("too_few_refinements")
    if len(refinements) > 5:
        refinements = refinements[:5]
    return refinements, truncated, errors


def strategist_reask(refinements: List[str], errors: List[str]) -> str:
    missing = ", ".join(errors) if errors else "format issues"
    return (
        "Your previous strategist output violated the contract. Return ONLY JSON: "
        '{"refinements":["..."], "truncated": false}. '
        f"Fix: {missing}. Provide 3–5 concise refinements (<=200 chars)."
    )


# ----------------------------- Worker -----------------------------
def parse_worker_sketch(text: str) -> Dict[str, Any]:
    """
    Try to parse a Worker JSON sketch. If not JSON, attempt to extract
    a sketch and key steps heuristically. Returns a dict with keys:
    {"sketch": str, "key_steps": List[str], "truncated": bool, "valid": bool, "source": str, "errors": List[str]}
    """
    out = {"sketch": "", "key_steps": [], "truncated": False, "valid": False, "source": "", "errors": []}
    if not (text or "").strip():
        out["errors"].append("empty_text")
        return out
    obj, source = parse_json_strict_or_repair(text)
    out["source"] = source
    if isinstance(obj, dict) and "sketch" in obj:
        try:
            out["sketch"] = str(obj.get("sketch", "")).strip()
            steps = obj.get("key_steps", [])
            if isinstance(steps, list):
                out["key_steps"] = [str(s).strip() for s in steps if str(s).strip()]
            out["truncated"] = bool(obj.get("truncated", False))
        except Exception as e:
            out["errors"].append(f"error:{e}")
    else:
        # Heuristic extraction: whole text as sketch; bullet lines as key steps
        cleaned = text.strip()
        out["sketch"] = cleaned
        bullets = re.findall(r"^[-*] +(.+)$", cleaned, flags=re.MULTILINE)
        out["key_steps"] = [b.strip() for b in bullets][:6]
        out["source"] = out["source"] or "heuristic"

    # Enforce constraints
    if len(out["sketch"]) < 120:
        out["errors"].append("sketch_too_short")
    if not (2 <= len(out["key_steps"]) <= 6):
        out["errors"].append("bad_key_steps_count")
    if any(len(s) > 140 for s in out["key_steps"]):
        out["errors"].append("step_too_long")
    out["valid"] = ("sketch_too_short" not in out["errors"]) and ("bad_key_steps_count" not in out["errors"]) and not out["truncated"]
    return out


def worker_reask(errors: List[str]) -> str:
    missing = ", ".join(errors) if errors else "contract violations"
    return (
        "Your previous Worker output violated the contract. Return ONLY JSON: "
        '{"sketch":"...","key_steps":["..."],"truncated":false}. '
        "Constraints: sketch >= 120 chars; 2–6 key_steps each <= 140 chars. "
        f"Fix: {missing}. Do not include any prose outside JSON."
    )


# ---------------------------- Verifier ----------------------------
def parse_verifier_json(text: str) -> Tuple[float, str, bool, str]:
    """
    Parse verifier output. Returns (score, reason, blocking, source)
    where source in {strict_json, repaired_json, json_fragment, loose_keys, regex_fallback, parse_failed}.
    """
    score = 0.0
    reason = ""
    blocking = False
    source = "parse_failed"
    if not (text or "").strip():
        return 0.5, "No response received from verifier", False, "empty_response_default"

    # 1) Strict or repaired JSON
    obj, src = parse_json_strict_or_repair(text)
    if isinstance(obj, dict):
        source = src
        try:
            score = float(obj.get("score", 0.0))
        except Exception:
            score = 0.0
        reason = str(obj.get("reason", "") or "").strip()[:200]
        blocking = bool(obj.get("blocking", False))
        score = max(0.0, min(1.0, score))
        return score, reason, blocking, source

    # 2) JSON-like fragment
    try:
        m = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            frag = m.group(0)
            try:
                data = json.loads(frag)
                score = float(data.get("score", 0.0))
                reason = str(data.get("reason", "") or "").strip()[:200]
                blocking = bool(data.get("blocking", False))
                score = max(0.0, min(1.0, score))
                return score, reason, blocking, "json_fragment"
            except Exception:
                # Loose keys
                sm = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', frag)
                rm = re.search(r'"reason"\s*:\s*"(.*?)"', frag, flags=re.DOTALL)
                bm = re.search(r'"blocking"\s*:\s*(true|false)', frag, flags=re.IGNORECASE)
                if sm:
                    score = max(0.0, min(1.0, float(sm.group(1))))
                    reason = (rm.group(1).strip() if rm else "").strip()[:200]
                    blocking = bool(bm and bm.group(1).lower() == "true")
                    return score, reason, blocking, "loose_keys"
    except Exception:
        pass

    # 3) Fallback any numeric 0..1 in text
    for mt in re.finditer(r"(\d+(?:\.\d+)?)", text):
        try:
            v = float(mt.group(1))
            if 0.0 <= v <= 1.0:
                return v, text[:100].strip(), False, "regex_fallback"
        except Exception:
            continue
    return 0.5, "Could not parse score from response, using default", False, "parse_failed"


def is_valid_for_adaptation(worker_valid: bool, verifier_source: str, score: float, blocking: bool, min_score: float = 0.0) -> bool:
    if blocking:
        return False
    if not worker_valid:
        return False
    if verifier_source in {"regex_fallback", "parse_failed", "empty_response_default"}:
        return False
    if not (0.0 <= score <= 1.0):
        return False
    if score < min_score:
        return False
    return True

