from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Any
import os

# Handle both package and script contexts
try:
    from .logging_utils import log_print
    from .nrpa import run_nrpa, NRPA_LEVELS, NRPA_ITER, NRPA_ALPHA, NRPA_MAX_DEPTH, NRPAConfig
except ImportError:
    try:
        from logging_utils import log_print
        from nrpa import run_nrpa, NRPA_LEVELS, NRPA_ITER, NRPA_ALPHA, NRPA_MAX_DEPTH, NRPAConfig
    except ImportError:
        # Fallback to relative imports as last resort
        from .logging_utils import log_print
        from .nrpa import run_nrpa, NRPA_LEVELS, NRPA_ITER, NRPA_ALPHA, NRPA_MAX_DEPTH, NRPAConfig


class StrategySelector:
    """Abstract base class for strategy selection."""

    def __init__(self, api_client_funcs: Dict[str, Callable], strategist_model_name: str):
        self.api_client_funcs = api_client_funcs
        self.strategist_model_name = strategist_model_name

    def select_strategy(
        self,
        problem_statement: str,
        other_prompts: List[str],
        system_prompt: str,
        telemetry=None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError


class SingleStrategySelector(StrategySelector):
    """Strategy selector that makes a single call to the CEO."""

    def __init__(self, api_client_funcs: Dict[str, Callable], strategist_model_name: str, meta_learner=None):
        super().__init__(api_client_funcs, strategist_model_name)
        # Single strategy selector doesn't use meta learner, but accepts for consistency

    def select_strategy(
        self,
        problem_statement: str,
        other_prompts: List[str],
        system_prompt: str,
        telemetry=None,
        **_: Any,
    ) -> str:
        log_print("[STRATEGY] Using single strategy selection.")
        strategist_payload = self.api_client_funcs["build_request_payload"](
            system_prompt=system_prompt,
            question_prompt=problem_statement,
            other_prompts=other_prompts,
        )
        strategist_response = self.api_client_funcs["send_api_request"](
            self.api_client_funcs["get_api_key"]("strategist"),
            strategist_payload,
            model_name=self.strategist_model_name,
            agent_type="strategist",
            telemetry=telemetry,
        )
        return self.api_client_funcs["extract_text_from_response"](strategist_response)


class NRPAStrategySelector(StrategySelector):
    """Strategy selector that runs the full NRPA loop with optional meta control."""

    def __init__(self, api_client_funcs: Dict[str, Callable], strategist_model_name: str, meta_learner=None):
        super().__init__(api_client_funcs, strategist_model_name)
        self.meta_learner = meta_learner

    def select_strategy(
        self,
        problem_statement: str,
        other_prompts: List[str],
        system_prompt: str,
        telemetry=None,
        enumerate_initial_strategies: Optional[Callable[[str, List[str]], List[str]]] = None,
        generate_refinements: Optional[Callable[[List[str], str, Dict[str, Any], Any], List[str]]] = None,
        run_strategic_simulation: Optional[Callable[[str, str, Any], str]] = None,
        lightweight_score_sketch: Optional[Callable[..., Tuple[float, str, str, bool]]] = None,
        **_: Any,
    ) -> str:
        log_print("[NRPA] Starting Strategist with NRPA Strategy Search...")
        if not (
            enumerate_initial_strategies
            and generate_refinements
            and run_strategic_simulation
            and lightweight_score_sketch
        ):
            raise ValueError("NRPAStrategySelector requires strategy generation and scoring functions")

        strategies = enumerate_initial_strategies(problem_statement, other_prompts)
        if not strategies:
            log_print("[FALLBACK] Switching to MCTS search")
            from .mcts import run_mcts_search
            return run_mcts_search(
                problem_statement,
                other_prompts,
                system_prompt,
                self.api_client_funcs,
                self.strategist_model_name,
                telemetry
            )

        # Persistent, file-backed cache configuration
        import threading
        import os
        import json
        import hashlib

        cache: Dict[str, Any] = {}
        cache["nrpa_usage"] = {"refine_calls": 0, "score_calls": 0}
        cache["lock"] = threading.Lock()

        cache_enabled = os.getenv("NRPA_CACHE_ENABLED", "1") in ("1", "true", "True", "yes", "YES")
        cache_dir = os.getenv("NRPA_CACHE_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache"))
        os.makedirs(cache_dir, exist_ok=True)

        def _problem_hash(text: str) -> str:
            return hashlib.md5(text.encode()).hexdigest()

        problem_h = _problem_hash(problem_statement)
        refine_store_path = os.path.join(cache_dir, f"refine_{problem_h}.json")
        score_store_path = os.path.join(cache_dir, f"score_{problem_h}.json")

        # Load stores if enabled
        refine_store: Dict[str, Any] = {}
        score_store: Dict[str, Any] = {}
        if cache_enabled:
            try:
                if os.path.exists(refine_store_path):
                    with open(refine_store_path, "r", encoding="utf-8") as f:
                        refine_store = json.load(f)
            except Exception:
                refine_store = {}
            try:
                if os.path.exists(score_store_path):
                    with open(score_store_path, "r", encoding="utf-8") as f:
                        score_store = json.load(f)
            except Exception:
                score_store = {}

        def children_provider(step: int, prefix: Tuple[str, ...]) -> List[str]:
            if step == 0:
                return strategies
            key = "|".join(prefix)
            if cache_enabled and key in refine_store:
                return list(refine_store.get(key, []))
            result = generate_refinements(list(prefix), problem_statement, cache, telemetry)
            if cache_enabled:
                with cache["lock"]:
                    # Limit entries to keep size reasonable
                    if len(refine_store) < 5000:
                        refine_store[key] = result
            # Budget accounting
            usage = cache.get("nrpa_usage")
            if usage is not None:
                usage["refine_calls"] = int(usage.get("refine_calls", 0)) + 1
            return result

        def score_fn(seq: List[str]) -> float:
            if not seq:
                return 0.0
            path_description = " -> ".join(seq)
            # ensure valid_adaptation map exists
            if cache.get("valid_adaptation") is None:
                cache["valid_adaptation"] = {}
            # Cache by path description
            if cache_enabled and path_description in score_store:
                score_reason = score_store[path_description]
                score = float(score_reason.get("score", 0.0))
                reason = str(score_reason.get("reason", ""))
                worker_valid = bool(score_reason.get("worker_valid", True))
                parsed_source = str(score_reason.get("source", "strict_json"))
                blocking = bool(score_reason.get("blocking", False))
            else:
                # Generate sketch and enforce Worker contract with one re-ask if needed
                from .io_contracts import parse_worker_sketch, worker_reask, is_valid_for_adaptation
                sketch_raw = run_strategic_simulation(path_description, problem_statement, telemetry)
                ws = parse_worker_sketch(sketch_raw)
                worker_valid = bool(ws.get("valid", False)) and not bool(ws.get("truncated", False))
                if telemetry:
                    try:
                        telemetry.record_worker_sketch(non_empty=bool(ws.get("sketch")), truncated=bool(ws.get("truncated", False)))
                        telemetry.record_parse(success=bool(ws.get("source") in ("strict_json", "repaired_json")))
                    except Exception:
                        pass
                if not worker_valid:
                    reask_msg = worker_reask(ws.get("errors", []))
                    sketch_raw = run_strategic_simulation(path_description, problem_statement, telemetry, reask_note=reask_msg)
                    ws = parse_worker_sketch(sketch_raw)
                    worker_valid = bool(ws.get("valid", False)) and not bool(ws.get("truncated", False))
                    if telemetry:
                        try:
                            telemetry.record_worker_sketch(non_empty=bool(ws.get("sketch")), truncated=bool(ws.get("truncated", False)))
                            telemetry.record_parse(success=bool(ws.get("source") in ("strict_json", "repaired_json")))
                        except Exception:
                            pass
                sketch_for_score = str(ws.get("sketch", "") or sketch_raw)
                score, reason, parsed_source, blocking = lightweight_score_sketch(sketch_for_score, telemetry)
                if telemetry:
                    try:
                        telemetry.record_verifier(valid=(parsed_source not in ("regex_fallback", "parse_failed", "empty_response_default")))
                    except Exception:
                        pass
                if cache_enabled:
                    with cache["lock"]:
                        if len(score_store) < 5000:
                            score_store[path_description] = {
                                "score": score,
                                "reason": reason,
                                "worker_valid": worker_valid,
                                "source": parsed_source,
                                "blocking": blocking,
                            }
            # Compute adaptation gating
            try:
                from .io_contracts import is_valid_for_adaptation as _is_valid
            except Exception:
                _is_valid = lambda *a, **k: True
            min_score = float(os.getenv("NRPA_ADAPT_MIN_SCORE", "0.6"))
            allow = _is_valid(worker_valid, parsed_source, score, blocking, min_score=min_score)
            cache["valid_adaptation"][path_description] = allow
            if telemetry:
                try:
                    telemetry.record_adaptation(high_confidence=allow)
                except Exception:
                    pass
            # Budget accounting
            usage = cache.get("nrpa_usage")
            if usage is not None:
                usage["score_calls"] = int(usage.get("score_calls", 0)) + 1
            log_print(f"[NRPA] Scored sequence: {path_description[:100]}... -> {score:.3f} ({reason[:50]})")
            return score

        # Build configuration (env-driven with defaults)
        config = NRPAConfig.from_env()

        log_print(
            f"[NRPA] Starting search: L={config.levels}, N={config.iterations}, Alpha={config.alpha}, MaxDepth={config.max_depth}, Temp={config.temperature}"
        )
        if telemetry:
            try:
                from .telemetry_ext import nrpa_start, nrpa_end  # package context
            except Exception:
                # Fallback when running as a script (no package context)
                from telemetry_ext import nrpa_start, nrpa_end

            nrpa_start(
                telemetry,
                {
                    "num_candidates": len(strategies),
                    "levels": config.levels,
                    "iterations": config.iterations,
                    "alpha": config.alpha,
                    "max_depth": config.max_depth,
                    "temperature": config.temperature,
                    "seed": config.seed,
                    "patience": config.patience,
                    "beam_width": config.beam_width,
                    "max_workers": config.max_workers,
                },
            )
        best_score, best_seq = run_nrpa(
            config=config,
            initial_strategies=strategies,
            children_provider=children_provider,
            score_fn=score_fn,
            cache=cache,
            telemetry=telemetry,
            meta_learner=self.meta_learner,  # NEW: Pass meta learner
        )
        # Persist a context snapshot for cross-session sharing
        try:
            from .context_store import ContextStore

            ctx = ContextStore.from_env()
            snapshot = {
                "problem_hash": hashlib.md5(problem_statement.encode()).hexdigest(),
                "num_candidates": len(strategies),
                "config": {
                    "levels": config.levels,
                    "iterations": config.iterations,
                    "alpha": config.alpha,
                    "max_depth": config.max_depth,
                    "temperature": config.temperature,
                    "seed": config.seed,
                    "patience": config.patience,
                    "max_seconds": config.max_seconds,
                    "max_calls": config.max_calls,
                    "beam_width": config.beam_width,
                    "max_workers": config.max_workers,
                },
                "usage": cache.get("nrpa_usage", {}),
                "best": {
                    "score": best_score,
                    "sequence": best_seq,
                    "chosen": " -> ".join(best_seq) if best_seq else (strategies[0] if strategies else ""),
                },
            }
            ctx.save_snapshot(snapshot)
        except Exception:
            pass
        # Flush persistent caches
        if cache_enabled:
            try:
                with open(refine_store_path, "w", encoding="utf-8") as f:
                    json.dump(refine_store, f, indent=2)
                with open(score_store_path, "w", encoding="utf-8") as f:
                    json.dump(score_store, f, indent=2)
            except Exception:
                pass
        if telemetry:
            try:
                from .telemetry_ext import nrpa_end
            except Exception:
                from telemetry_ext import nrpa_end

            nrpa_end(telemetry, {"best_score": best_score})
        chosen = " -> ".join(best_seq) if best_seq else (strategies[0] if strategies else "Direct Approach: Solve the problem directly using standard techniques")
        log_print(f"[NRPA] Best sequence (score={best_score:.3f}): {chosen}")
        return chosen
