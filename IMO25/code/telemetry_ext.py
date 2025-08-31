"""
telemetry_ext.py

Non-invasive telemetry helpers for recording NRPA lifecycle events.

Purpose
- Provide thin wrappers that annotate the TelemetrySystem (defined in agent.py) with NRPA-specific events
  without importing or tightly coupling to the TelemetrySystem class.
- Keep agent.py clean by centralizing event naming and payload formatting for:
    * NRPA_START: initial candidate summary (e.g., number of strategies)
    * NRPA_ITERATION: per-iteration selection, score, and policy summary
    * NRPA_END: final chosen strategy and policy state

Why separate?
- Decouples logging/telemetry responsibilities from core search and orchestration logic.
- Allows alternate telemetry systems to be injected without code changes here; wrappers no-op if telemetry is None.

Usage
- Called from agent.init_explorations() around the NRPA loop.

Resilience
- Each function guards against missing telemetry or logging failures to avoid breaking the solve loop.
"""
from __future__ import annotations
from typing import Any, Dict, List


# Lightweight wrappers to record NRPA-related telemetry without modifying the core TelemetrySystem class.
# The TelemetrySystem in agent.py exposes:
#   - log_event(event_type: str, description: str)
#   - metrics dict and save_metrics()
#
# These helpers simply format and forward NRPA events via log_event.


def nrpa_start(telemetry: Any, summary: Dict[str, Any]) -> None:
    """
    Log NRPA session start with an initial summary payload.
    Example summary: {"num_candidates": 5}
    """
    if telemetry is None:
        return
    try:
        telemetry.log_event("NRPA_START", f"Start NRPA: {summary}")
    except Exception:
        # Be resilient to formatting/logging failures
        pass


def nrpa_iteration(telemetry: Any, iteration: int, data: Dict[str, Any]) -> None:
    """
    Log one iteration of NRPA with selection, score, and policy summaries.
    Example data: {"selected_path": "...", "score": 0.63, "reason": "...", "policy": {...}}
    """
    if telemetry is None:
        return
    try:
        payload = {"iteration": iteration, **data}
        telemetry.log_event("NRPA_ITERATION", f"Iter {iteration}: {payload}")
    except Exception:
        pass


def nrpa_end(telemetry: Any, result: Dict[str, Any]) -> None:
    """
    Log NRPA session end with final best strategy and policy state.
    Example result: {"chosen": "Inversion: ...", "policy": {...}}
    """
    if telemetry is None:
        return
    try:
        telemetry.log_event("NRPA_END", f"End NRPA: {result}")
    except Exception:
        pass


# Meta Controller Telemetry Functions

def meta_decision(telemetry: Any, state_key: str, action: str, q_value: float,
                  exploration_state: Dict[str, Any]) -> None:
    """
    Log meta controller exploration decisions.

    Args:
        state_key: Hash of the current exploration state
        action: Decision made (continue, stop_eval, switch, abandon)
        q_value: Q-value for the chosen action
        exploration_state: Current exploration context
    """
    if telemetry is None:
        return
    try:
        payload = {
            "state_key": state_key,
            "action": action,
            "q_value": q_value,
            "strategy_path": exploration_state.get("strategy_path", []),
            "current_depth": exploration_state.get("current_depth", 0),
            "partial_score": exploration_state.get("partial_score", 0.0)
        }
        telemetry.log_event("META_DECISION", f"Meta decision: {payload}")
    except Exception:
        pass


def meta_learning(telemetry: Any, state_key: str, action: str,
                 old_q: float, new_q: float, reward: float) -> None:
    """
    Log Q-learning updates and learning progress.

    Args:
        state_key: Hash of the state being updated
        action: Action whose Q-value was updated
        old_q: Q-value before update
        new_q: Q-value after update
        reward: Reward received for this update
    """
    if telemetry is None:
        return
    try:
        payload = {
            "state_key": state_key,
            "action": action,
            "old_q": old_q,
            "new_q": new_q,
            "reward": reward,
            "improvement": new_q - old_q
        }
        telemetry.log_event("META_LEARNING", f"Q-learning update: {payload}")
    except Exception:
        pass


def meta_stats(telemetry: Any, stats: Dict[str, Any]) -> None:
    """
    Log meta controller statistics and performance metrics.

    Args:
        stats: Statistics from meta learner (total_states, avg_q_value, etc.)
    """
    if telemetry is None:
        return
    try:
        telemetry.log_event("META_STATS", f"Meta controller stats: {stats}")
    except Exception:
        pass


def exploration_efficiency(telemetry: Any, metrics: Dict[str, Any]) -> None:
    """
    Log exploration efficiency metrics comparing meta-controlled vs standard NRPA.

    Args:
        metrics: Efficiency comparison metrics
    """
    if telemetry is None:
        return
    try:
        telemetry.log_event("EXPLORATION_EFFICIENCY", f"Efficiency metrics: {metrics}")
    except Exception:
        pass


def early_stopping_event(telemetry: Any, reason: str, strategy_path: List[str],
                        final_score: float, steps_saved: int) -> None:
    """
    Log early stopping events to track efficiency improvements.

    Args:
        reason: Why exploration was stopped early
        strategy_path: Strategy path at stopping point
        final_score: Final score achieved
        steps_saved: Estimated steps saved by early stopping
    """
    if telemetry is None:
        return
    try:
        payload = {
            "reason": reason,
            "strategy_path": strategy_path,
            "final_score": final_score,
            "steps_saved": steps_saved,
            "path_length": len(strategy_path)
        }
        telemetry.log_event("EARLY_STOPPING", f"Early stop: {payload}")
    except Exception:
        pass
