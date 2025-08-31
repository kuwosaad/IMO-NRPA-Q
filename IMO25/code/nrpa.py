"""
nrpa.py

Nested Rollout Policy Adaptation for strategy refinement search.

Implements the NRPA algorithm atop our strategy enumeration and refinement pipeline:
- States are (step, prefix) tuples representing a refinement path
- Actions are refinements of the current strategy prefix
- Policy adapts via gradient ascent on rollout sequences
- Rollouts use softmax policy to bias future trials

Key functions:
- run_nrpa: main entrypoint, returns best (score, seq)
- rollout: simulate a path using current policy
- adapt: shift policy weights toward a successful sequence
- generate_refinements: use LLM to refine strategy prefix
"""
import os
import json
import hashlib
import math
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Callable, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
# Backwards-compatible module-level defaults (preserved for external imports if any)
NRPA_LEVELS = int(os.getenv("NRPA_LEVELS", 2))
NRPA_ITER = int(os.getenv("NRPA_ITER", 60))
NRPA_ALPHA = float(os.getenv("NRPA_ALPHA", 1.0))
NRPA_MAX_DEPTH = int(os.getenv("NRPA_MAX_DEPTH", 4))


@dataclass
class NRPAConfig:
    """Explicit configuration for NRPA to improve reproducibility and control."""
    levels: int
    iterations: int
    alpha: float
    max_depth: int
    temperature: float = 1.0
    seed: Optional[int] = None
    patience: int = 0
    max_seconds: Optional[int] = None
    max_calls: Optional[int] = None
    beam_width: int = 1
    max_workers: int = 1

    # Meta controller settings (Phase 2)
    use_meta_control: bool = True
    meta_learner_path: Optional[str] = None
    early_stop_threshold: float = 0.7
    exploration_penalty: float = -0.1
    efficiency_reward: float = 0.1

    @staticmethod
    def from_env() -> "NRPAConfig":
        # Importing config ensures dotenv is loaded
        try:
            from config import _config  # noqa: F401
        except Exception:
            pass
        return NRPAConfig(
            levels=int(os.getenv("NRPA_LEVELS", NRPA_LEVELS)),
            iterations=int(os.getenv("NRPA_ITER", NRPA_ITER)),
            alpha=float(os.getenv("NRPA_ALPHA", NRPA_ALPHA)),
            max_depth=int(os.getenv("NRPA_MAX_DEPTH", NRPA_MAX_DEPTH)),
            temperature=float(os.getenv("NRPA_TEMPERATURE", 1.0)),
            seed=(int(os.getenv("NRPA_SEED", "")) if os.getenv("NRPA_SEED") else None),
            patience=int(os.getenv("NRPA_PATIENCE", 0)),
            max_seconds=(int(os.getenv("NRPA_MAX_SECONDS", "")) if os.getenv("NRPA_MAX_SECONDS") else None),
            max_calls=(int(os.getenv("NRPA_MAX_CALLS", "")) if os.getenv("NRPA_MAX_CALLS") else None),
            beam_width=int(os.getenv("NRPA_BEAM_WIDTH", 1)),
            max_workers=int(os.getenv("NRPA_MAX_WORKERS", 1)),
            # Meta controller settings
            use_meta_control=os.getenv("NRPA_USE_META_CONTROL", "1") in ("1", "true", "True", "yes", "YES"),
            meta_learner_path=os.getenv("NRPA_META_LEARNER_PATH"),
            early_stop_threshold=float(os.getenv("NRPA_EARLY_STOP_THRESHOLD", "0.7")),
            exploration_penalty=float(os.getenv("NRPA_EXPLORATION_PENALTY", "-0.1")),
            efficiency_reward=float(os.getenv("NRPA_EFFICIENCY_REWARD", "0.1")),
        )

    def seed_everything(self) -> None:
        if self.seed is None:
            return
        random.seed(self.seed)
        try:
            import numpy as _np  # type: ignore

            _np.random.seed(self.seed)
        except Exception:
            pass

# --- POLICY UTILS ---
Code = str  # hashable representation of action
Policy = Dict[Code, float]

def logsumexp(xs: List[float]) -> float:
    """Numerically stable log(sum(exp(x)))"""
    if not xs:
        return float('-inf')
    max_x = max(xs)
    if max_x == float('-inf'):
        return float('-inf')
    return max_x + math.log(sum(math.exp(x - max_x) for x in xs))

class PolicyManager:
    """Manages the policy dictionary and related operations."""

    def __init__(self, max_weight: float = 100.0, temperature: float = 1.0):
        self.policy: Policy = defaultdict(float)
        self.max_weight = max_weight
        self.temperature = max(0.0, float(temperature))

    def softmax_sample(self, actions: List[str]) -> str:
        """Sample action proportional to exp(pol[code(a)] / temperature)."""
        if not actions:
            raise ValueError("No actions to sample")
        if self.temperature <= 0:
            # Argmax behavior
            return max(actions, key=lambda a: self.policy.get(code(a), 0.0))
        codes = [code(a) for a in actions]
        weights = [math.exp(self.policy.get(c, 0.0) / self.temperature) for c in codes]
        total = sum(weights)
        if total <= 0:
            return random.choice(actions)
        probs = [w / total for w in weights]
        return random.choices(actions, weights=probs, k=1)[0]
    
    def adapt(self, seq: List[str], alpha: float, children_provider: Callable[[int, Tuple[str, ...]], List[str]]) -> None:
        """
        Adapt policy weights toward a successful sequence using gradient ascent.
        
        For each action a_t in the sequence:
        - Increase pol[code(a_t)] by alpha
        - Decrease all pol[code(a)] by alpha * P(a|s_t) to maintain normalization
        """
        prefix = ()
        for step, action in enumerate(seq):
            legal_actions = children_provider(step, prefix)
            if not legal_actions:
                break
                
            action_code = code(action)
            self.policy[action_code] = self.policy.get(action_code, 0.0) + alpha
            
            # Compute normalization constant z = sum_a exp(pol[code(a)])
            codes = [code(a) for a in legal_actions]
            log_z = logsumexp([self.policy.get(c, 0.0) for c in codes])
            
            # Subtract alpha * P(a|s) from each action
            for a, c in zip(legal_actions, codes):
                prob = math.exp(self.policy.get(c, 0.0) - log_z)
                self.policy[c] = self.policy.get(c, 0.0) - alpha * prob
                
            prefix = prefix + (action,)
            
        self.clip_policy()
    
    def clip_policy(self) -> None:
        """Clip policy weights to prevent overflow"""
        for k, v in self.policy.items():
            self.policy[k] = max(-self.max_weight, min(self.max_weight, v))

# --- STATE/ACTION CODES ---
def code(action: str) -> Code:
    """Generate a unique, deterministic code for an action"""
    return hashlib.md5(action.encode()).hexdigest()

# --- CORE NRPA ---

from .trace import trace

@trace
def rollout(
    policy_manager: PolicyManager,
    initial_candidates: List[str],
    children_provider: Callable[[int, Tuple[str, ...]], List[str]],
    score_fn: Callable[[List[str]], float],
    cache: Dict[str, Any],
    config: NRPAConfig,
    meta_learner=None,  # NEW: Optional meta controller
) -> Tuple[float, List[str]]:
    """
    Perform a single rollout using the current policy with optional meta control.

    Args:
        meta_learner: Optional ExplorationMetaLearner for intelligent exploration control

    Returns:
        (score, sequence) where sequence is the path taken
    """
    seq = []
    prefix = ()

    # Start with initial candidates
    if initial_candidates:
        action = policy_manager.softmax_sample(initial_candidates)
        seq.append(action)
        prefix = (action,)

    # NEW: Meta controller integration
    if meta_learner and config.use_meta_control:
        return enhanced_rollout_with_meta_control(
            policy_manager, initial_candidates, children_provider,
            score_fn, cache, config, meta_learner
        )

    # Original NRPA rollout logic
    # Refine up to max_depth
    for step in range(1, config.max_depth):
        legal_actions = children_provider(step, prefix)
        if not legal_actions:
            break
        action = policy_manager.softmax_sample(legal_actions)
        seq.append(action)
        prefix = prefix + (action,)

    # Score the final sequence
    score = score_fn(seq)
    return score, seq

def enhanced_rollout_with_meta_control(
    policy_manager: PolicyManager,
    initial_candidates: List[str],
    children_provider: Callable[[int, Tuple[str, ...]], List[str]],
    score_fn: Callable[[List[str]], float],
    cache: Dict[str, Any],
    config: NRPAConfig,
    meta_learner,
) -> Tuple[float, List[str]]:
    """
    Enhanced rollout with Q-learning meta controller for intelligent exploration decisions.
    """
    seq = []
    prefix = ()

    # Start with initial candidates
    if initial_candidates:
        action = policy_manager.softmax_sample(initial_candidates)
        seq.append(action)
        prefix = (action,)

    # Progressive refinement with meta control
    for step in range(1, config.max_depth):
        # Get available actions
        legal_actions = children_provider(step, prefix)
        if not legal_actions:
            break

        # Extract problem features for meta learner (simplified)
        problem_features = cache.get("problem_features", {"type": "unknown", "complexity": "medium"})

        # Create exploration state for meta decision
        try:
            from .strategy_scorer import quick_strategy_score
            from .exploration_meta import ExplorationAction, ExplorationState
        except ImportError:
            # Fallback if modules not available
            def quick_strategy_score(path, problem):
                return 0.5  # Neutral score
            class ExplorationAction:
                STOP_AND_EVALUATE = "stop_eval"
            @dataclass
            class ExplorationState:
                strategy_path: List[str]
                current_depth: int
                partial_score: float
                problem_features: Dict[str, Any]
                def to_key(self) -> str: return ""

        partial_score = quick_strategy_score(seq, cache.get("problem_statement", ""))

        exploration_state = ExplorationState(
            strategy_path=seq,
            current_depth=step,
            partial_score=partial_score,
            problem_features=problem_features
        )

        # Ask meta learner what to do (with early stopping consideration)
        if hasattr(meta_learner, 'should_stop_early') and meta_learner.should_stop_early(exploration_state, config.early_stop_threshold):
            meta_decision = ExplorationAction.STOP_AND_EVALUATE
            # Log early stopping event
            try:
                from .telemetry_ext import early_stopping_event
                early_stopping_event(cache.get("telemetry"), "threshold_exceeded",
                                   seq, 0.0, config.max_depth - step)
            except Exception:
                pass
        else:
            meta_decision = meta_learner.decide_exploration(exploration_state, legal_actions, cache.get("telemetry"))

        # Act based on meta decision
        if meta_decision == ExplorationAction.STOP_AND_EVALUATE:
            # Stop exploring, evaluate current path
            final_score = score_fn(seq)
            # Learn from this decision
            meta_learner.learn_from_outcome(
                exploration_state.to_key(),
                meta_decision,
                config.efficiency_reward,  # Reward for efficient stopping
                final_score,
                cache.get("telemetry")
            )
            return final_score, seq

        elif meta_decision == ExplorationAction.ABANDON_PATH:
            # This path is not promising, return low score
            meta_learner.learn_from_outcome(
                exploration_state.to_key(),
                meta_decision,
                config.exploration_penalty,  # Penalty for wasted exploration
                0.0,
                cache.get("telemetry")
            )
            return 0.0, seq

        elif meta_decision == ExplorationAction.SWITCH_BRANCH:
            # Try different refinement - override NRPA policy
            if len(legal_actions) > 1:
                # Select alternative action (simple heuristic: pick different from current)
                current_action = seq[-1] if seq else ""
                alternative_actions = [a for a in legal_actions if a != current_action]
                if alternative_actions:
                    action = random.choice(alternative_actions)
                else:
                    action = policy_manager.softmax_sample(legal_actions)
            else:
                action = policy_manager.softmax_sample(legal_actions)
        else:  # CONTINUE_EXPLORING
            # Normal NRPA behavior
            action = policy_manager.softmax_sample(legal_actions)
            
            # --- REWARD SHAPING ---
            # Provide an intermediate reward based on the quality of the partial path
            intermediate_reward = partial_score * 0.1  # Scale to be smaller than final reward
            meta_learner.learn_from_outcome(
                exploration_state.to_key(),
                meta_decision,
                intermediate_reward,
                0.0,  # No final score yet
                cache.get("telemetry"),
                is_intermediate=True
            )

        seq.append(action)
        prefix = prefix + (action,)

    # Reached max depth or no more actions
    final_score = score_fn(seq)
    return final_score, seq

def run_nrpa(
    config: NRPAConfig,
    initial_strategies: List[str],
    children_provider: Callable[[int, Tuple[str, ...]], List[str]],
    score_fn: Callable[[List[str]], float],
    cache: Dict[str, Any],
    telemetry: Any = None,
    meta_learner=None,  # NEW: Optional meta controller
) -> Tuple[float, List[str]]:
    """
    Run NRPA to find the best strategy sequence.
    
    Args:
        levels: Nesting depth (0 = base rollout)
        iterations: Number of iterations per level
        alpha: Policy update step size
        initial_strategies: Root-level strategy candidates
        children_provider: Function to generate refinements for a state
        score_fn: Function to score a complete sequence
        cache: Shared cache for LLM results
        
    Returns:
        (best_score, best_sequence)
    """
    # Seed reproducibility at the start of the outermost call
    if cache.get("_nrpa_seeded") is not True:
        config.seed_everything()
        cache["_nrpa_seeded"] = True
        if telemetry is not None:
            try:
                from telemetry_ext import nrpa_start as _nrpa_start

                _nrpa_start(
                    telemetry,
                    {
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
                )
            except Exception:
                pass

    def compute_policy_introspection(policy_manager: PolicyManager) -> Dict[str, Any]:
        """Compute top-k and entropy up to first 3 steps without mutating state."""
        summary: Dict[str, Any] = {"steps": []}
        prefix: Tuple[str, ...] = ()
        max_steps = min(3, config.max_depth)
        for step in range(0, max_steps):
            try:
                legal = children_provider(step, prefix)
            except Exception:
                legal = []
            if not legal:
                break
            codes = [code(a) for a in legal]
            weights = [policy_manager.policy.get(c, 0.0) for c in codes]
            if policy_manager.temperature <= 0:
                probs = [1.0 if w == max(weights) else 0.0 for w in weights]
            else:
                exps = [math.exp(w / policy_manager.temperature) for w in weights]
                total = sum(exps) or 1.0
                probs = [x / total for x in exps]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            # top-k by raw weight
            topk = sorted(zip(legal, weights), key=lambda t: t[1], reverse=True)[:3]
            summary["steps"].append({
                "step": step,
                "prefix": " -> ".join(prefix),
                "topk": [{"action": a, "weight": float(w)} for a, w in topk],
                "entropy": float(entropy),
            })
            # descend along current best action for introspection only
            prefix = prefix + (topk[0][0],)
        return summary

    start_time = time.time()

    if config.levels == 0:
        policy_manager = PolicyManager(temperature=config.temperature)
        # Optional beam at base level
        if config.beam_width > 1 and config.max_workers > 1:
            sequences: List[List[str]] = []
            for _ in range(config.beam_width):
                # Sample sequence without scoring (score later in parallel)
                seq = []
                prefix: Tuple[str, ...] = ()
                if initial_strategies:
                    action = policy_manager.softmax_sample(initial_strategies)
                    seq.append(action)
                    prefix = (action,)
                for step in range(1, config.max_depth):
                    legal_actions = children_provider(step, prefix)
                    if not legal_actions:
                        break
                    action = policy_manager.softmax_sample(legal_actions)
                    seq.append(action)
                    prefix = prefix + (action,)
                sequences.append(seq)

            # Score in parallel
            lock = cache.get("lock")
            def safe_score(s: List[str]) -> Tuple[float, List[str]]:
                # No mutation here, just compute score
                # If the score_fn uses cache, ensure thread-safe writes
                if lock is not None:
                    # Score function itself should manage its own locking when mutating cache
                    pass
                return score_fn(s), s

            with ThreadPoolExecutor(max_workers=config.max_workers) as ex:
                results = list(ex.map(safe_score, sequences))
            best_score, best_seq = max(results, key=lambda t: t[0])
            return best_score, best_seq
        else:
            return rollout(policy_manager, initial_strategies, children_provider, score_fn, cache, config, meta_learner)

    best_score = float("-inf")
    best_seq: List[str] = []
    policy_manager = PolicyManager(temperature=config.temperature)

    no_improve_streak = 0
    for iteration in range(config.iterations):
        # Budgets/time checks
        if config.max_seconds is not None and (time.time() - start_time) > config.max_seconds:
            if telemetry is not None:
                try:
                    from telemetry_ext import nrpa_end as _nrpa_end

                    _nrpa_end(telemetry, {"stop_reason": "time", "best_score": best_score, "best_seq": best_seq})
                except Exception:
                    pass
            break
        if config.max_calls is not None:
            usage = cache.get("nrpa_usage") or {}
            used_calls = int(usage.get("refine_calls", 0)) + int(usage.get("score_calls", 0))
            if used_calls >= config.max_calls:
                if telemetry is not None:
                    try:
                        from telemetry_ext import nrpa_end as _nrpa_end

                        _nrpa_end(telemetry, {"stop_reason": "calls", "best_score": best_score, "best_seq": best_seq})
                    except Exception:
                        pass
                break

        score, seq = run_nrpa(config.__class__(
            levels=config.levels - 1,
            iterations=config.iterations,
            alpha=config.alpha,
            max_depth=config.max_depth,
            temperature=config.temperature,
            seed=config.seed,
            patience=config.patience,
            max_seconds=(None if config.max_seconds is None else max(0, config.max_seconds - int(time.time() - start_time))),
            max_calls=config.max_calls,
            beam_width=config.beam_width,
            max_workers=config.max_workers,
            # NEW: Pass meta controller settings
            use_meta_control=config.use_meta_control,
            meta_learner_path=config.meta_learner_path,
            early_stop_threshold=config.early_stop_threshold,
            exploration_penalty=config.exploration_penalty,
            efficiency_reward=config.efficiency_reward,
        ), initial_strategies, children_provider, score_fn, cache, telemetry, meta_learner)

        last_score = score
        if score >= best_score:
            improved = score > best_score
            best_score = score
            best_seq = seq
            # Adaptation gating based on validation flags stored by score_fn
            try:
                path_desc = " -> ".join(seq)
                allow = True
                valid_map = cache.get("valid_adaptation") or {}
                if path_desc in valid_map:
                    allow = bool(valid_map.get(path_desc, True))
                if allow:
                    policy_manager.adapt(seq, config.alpha, children_provider)
                if telemetry is not None:
                    try:
                        telemetry.record_adaptation(high_confidence=allow)
                    except Exception:
                        pass
            except Exception:
                # Fail-closed: if any issue, skip adaptation for safety
                pass
            no_improve_streak = 0 if improved else (no_improve_streak + 1)
        else:
            no_improve_streak += 1

        # Telemetry per-iteration
        if telemetry is not None:
            try:
                from telemetry_ext import nrpa_iteration as _nrpa_iteration

                policy_summary = compute_policy_introspection(policy_manager)
                _nrpa_iteration(
                    telemetry,
                    iteration,
                    {
                        "best_score_so_far": best_score,
                        "last_score": last_score,
                        "policy": policy_summary,
                    },
                )
            except Exception:
                pass

        # Early stopping on plateau
        if config.patience > 0 and no_improve_streak >= config.patience:
            if telemetry is not None:
                try:
                    from telemetry_ext import nrpa_end as _nrpa_end

                    _nrpa_end(telemetry, {"stop_reason": "patience", "best_score": best_score, "best_seq": best_seq})
                except Exception:
                    pass
            break

    return best_score, best_seq
