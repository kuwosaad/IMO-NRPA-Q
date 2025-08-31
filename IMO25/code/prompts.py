"""
prompts.py

Prompt templates and robust parsers for the CEO-Genius and NRPA pipeline.

Overview
- Defines the Strategist enumeration prompt that asks for 3–5 concise, labeled strategies.
- Defines the Worker sketch prompt used for cheap simulations during NRPA iterations.
- Defines the Lightweight Verifier prompt that returns a strict JSON viability score.
- Provides parser helpers that are resilient to model formatting drift (list parsing, JSON extraction).

Design Goals
- Deterministic, structured prompts to stabilize downstream parsing.
- Strict output contracts where useful (JSON scoring), with fallback parsing to prevent crashes.
- Minimal surface area so other modules (agent.py) can import and use without additional logic.

Usage
- STRATEGIST_ENUM_PROMPT: called by enumerate_initial_strategies in agent.py
- WORKER_SKETCH_PROMPT: used by run_strategic_simulation for short rollouts
- LIGHTWEIGHT_VERIFIER_PROMPT: used by lightweight_score_sketch to get {"score", "reason"}
- parse_strategies_list, parse_viability_score: tolerant parsing functions feeding NRPA loop.
"""
# Prompt templates and lightweight scoring utilities for NRPA
from __future__ import annotations

from typing import Any, Dict

# Import trace decorator
try:
    from .trace import trace
except ImportError:
    # Fallback for direct execution
    def trace(func):
        return func


def _brace_escape(val: str) -> str:
    return val.replace("{", "{{").replace("}", "}}")


def safe_format(template: str, **kwargs: Dict[str, Any]) -> str:
    """
    Safely format a template that may contain literal JSON braces by:
    - Escaping all braces in the template
    - Un-escaping placeholders for provided keys only
    - Escaping braces in values to prevent accidental placeholders
    """
    # Escape entire template
    tmp = template.replace("{", "{{").replace("}", "}}")
    # Unescape placeholders for provided keys
    for k in kwargs.keys():
        tmp = tmp.replace("{{" + k + "}}", "{" + k + "}")
    # Escape braces in values
    esc_kwargs: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, str):
            esc_kwargs[k] = _brace_escape(v)
        else:
            esc_kwargs[k] = v
    return tmp.format(**esc_kwargs)

# Strategist prompt to enumerate 3–5 high-level strategies using Strategic Canvas
STRATEGIST_ENUM_PROMPT = """
You are an IMO problem strategist ("CEO"). You must approach this problem with rigorous first-principles thinking and adversarial analysis.

First, deconstruct the problem while identifying vulnerabilities by creating a Strategic Canvas:

Problem Type: Identify the core mathematical domain
Key Objects: List mathematical entities involved
Constraints: Enumerate all conditions and restrictions
Goal State: Precisely state what must be found/proven
Attack Surfaces: Identify 2-3 critical vulnerabilities in potential approaches (e.g., "Unverified assumptions about base cases", "Gaps in asymptotic bounds")

Only after this analysis, propose 3–5 distinct strategic paths that directly reference these components and mitigate identified attack surfaces.
Each strategy must be a concise one-liner starting with a label and colon, explicitly addressing vulnerabilities.

Examples:
"Combinatorial Enumeration: Systematically count configurations of [Key Objects] under [Constraints] to characterize [Goal State] - mitigates [Attack Surface] by exhaustive verification"
"Geometric Construction: Build explicit examples satisfying [Constraints] - avoids [Attack Surface] by concrete demonstration"

Return ONLY a JSON object with this exact structure:
{
  "canvas": {
    "problem_type": "string",
    "key_objects": ["string"],
    "constraints": ["string"],
    "goal_state": "string",
    "attack_surfaces": ["string"]  # New field
  },
  "strategies": ["string"]
}

Example response:
{
  "canvas": {
    "problem_type": "Functional Inequality",
    "key_objects": ["bonza function f"],
    "constraints": ["f(a) | b^a - f(b)^{f(a)}"],
    "goal_state": "Find minimal c such that f(n) ≤ cn",
    "attack_surfaces": ["Unvalidated assumptions about f(2)", "Asymptotic extrapolation without error bounds"]
  },
  "strategies": [
    "Divisibility Exploitation: Fix a and vary b to bound f(a) - validates f(2) via small n checks",
    "Extremal Construction: Identify maximal f(n)/n sequence - avoids asymptotic gaps by explicit computation"
  ]
}

Problem:
{problem_statement}
"""

# Strategy refinement prompt for NRPA (strict JSON object)
STRATEGY_REFINEMENT_PROMPT = """
You are an IMO strategist. Given a partial strategy path, propose 3–5 specific refinements or next steps with rigorous mathematical grounding.

The current path is:
{path_prefix}

Contract:
Return ONLY a strict JSON object with this exact shape and nothing else:
{"refinements": ["string", ...], "truncated": false}
- Provide 3–5 unique, concise refinements (<= 200 characters each).
- No prose outside JSON. If you cannot provide 3 items, set "truncated": true.
"""

# Worker prompt to produce a brief, high-level proof sketch under constraints
WORKER_SKETCH_PROMPT = """
You are an IMO "Worker" asked to perform a short, targeted simulation for a strategic path.
You must approach this with rigorous mathematical thinking, not hand-waving.

Contract (return ONLY JSON and nothing else; end with ###END###):
{"sketch": "...", "key_steps": ["..."], "truncated": false}
Constraints:
- sketch must be >= 120 characters (hard minimum) and concise.
- key_steps must contain 2–6 bullet strings; each <= 140 characters.
- If you run out of space, set truncated=true.

Problem:
{problem_statement}

Selected Strategic Path:
{path_description}

Return only the JSON object described above, no extra commentary. ###END###
"""

# Lightweight verifier prompt to score viability of a sketch
LIGHTWEIGHT_VERIFIER_PROMPT = """
Score this sketch from 0.0 to 1.0 based on progress toward solution:
- 0.0: No progress (problem restatement)
- 0.2: Partial progress with minor flaws
- 0.5: Solid progress with correct core approach
- 0.8: Near-complete solution path
- 1.0: Complete, rigorous solution path

Reward partial progress and penalize only critical flaws. Explain in one sentence (<= 200 chars). If sketch is completely invalid, set blocking=true.

Return ONLY JSON:
{"score": 0.5, "reason": "justification", "blocking": false}

Sketch:
{sketch}
"""

# Fallback parser hints for extracting list items or JSON-like content
@trace
def parse_strategies_list(text: str) -> list[str]:
    """
    Extract 3–5 strategy lines from text with resilience to format drift.
    
    Now expects JSON object with canvas and strategies from updated STRATEGIST_ENUM_PROMPT.
    If JSON parsing fails, falls back to default strategies.

    Returns:
      A list of up to 5 strategy strings.
    """
    import json
    import re
    text = text or ""
    
    # Handle completely empty responses
    if not text.strip():
        return ["Direct Approach: Solve the problem directly using standard techniques",
                "Case Analysis: Break the problem into manageable cases",
                "Proof by Contradiction: Assume the opposite and derive a contradiction"]

    # 1. Strip markdown fences
    text = re.sub(r"^\s*```(json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)

    # 2. Attempt to parse the entire string as JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "strategies" in data:
            strategies = data.get("strategies", [])
            if isinstance(strategies, list) and all(isinstance(s, str) for s in strategies):
                # Filter to reasonable length strategies
                filtered = [s for s in strategies if 1 <= len(s) <= 300] # Increased max length
                if filtered:
                    return filtered[:5]
    except json.JSONDecodeError:
        # JSON was invalid. Let's try to find a JSON object inside the text.
        # This can happen if the LLM adds extra text before or after the JSON.
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict) and "strategies" in data:
                    strategies = data.get("strategies", [])
                    if isinstance(strategies, list) and all(isinstance(s, str) for s in strategies):
                        filtered = [s for s in strategies if 1 <= len(s) <= 300]
                        if filtered:
                            return filtered[:5]
            except json.JSONDecodeError:
                # The extracted substring was also not valid JSON.
                pass # Fall through to default

    # 3. If all parsing fails, return default strategies
    return ["Direct Approach: Solve the problem directly using standard techniques",
            "Case Analysis: Break the problem into manageable cases",
            "Proof by Contradiction: Assume the opposite and derive a contradiction"]


def parse_viability_score(text: str) -> tuple[float, str, str, bool]:
    """
    Parse verifier output using io_contracts.parse_verifier_json.

    Returns (score, reason, source, blocking)
    """
    from .io_contracts import parse_verifier_json
    score, reason, blocking, source = parse_verifier_json(text or "")
    return score, reason, source, blocking
