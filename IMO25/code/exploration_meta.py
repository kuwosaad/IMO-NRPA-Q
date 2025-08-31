"""
exploration_meta.py

Q-Learning meta controller for NRPA exploration guidance.
Provides intelligent stopping decisions during strategy rollout.
"""

from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import os
import math
import random
from collections import defaultdict
from .strategy_scorer import extract_strategy_features

class ExplorationAction(Enum):
    CONTINUE_EXPLORING = "continue"
    STOP_AND_EVALUATE = "stop_eval"
    SWITCH_BRANCH = "switch"
    ABANDON_PATH = "abandon"

@dataclass
class ExplorationState:
    strategy_path: List[str]
    current_depth: int
    partial_score: float
    problem_features: Dict[str, Any]

    def to_key(self) -> str:
        """Generates a key for the Q-table, now using abstracted strategy features."""
        strategy_features = extract_strategy_features(self.strategy_path)
        
        # Create a sorted string from strategy features for a consistent hash
        strategy_feature_str = "|".join(f"{k}:{v}" for k, v in sorted(strategy_features.items()))
        
        # Combine with other state components
        problem_features_hash = hash(json.dumps(self.problem_features, sort_keys=True))
        
        # Discretize partial_score to reduce state space size
        discretized_score = round(self.partial_score * 10) / 10
        
        return f"{strategy_feature_str}|{self.current_depth}|{discretized_score}|{problem_features_hash}"

class ExplorationMetaLearner:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.2, exploration_strategy='epsilon_greedy'):
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_steps = 0
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.exploration_strategy = exploration_strategy

    def decide_exploration(self, state: ExplorationState,
                          available_actions: List[str], telemetry=None) -> ExplorationAction:
        """Core decision-making method using the configured exploration strategy."""
        self.total_steps += 1
        state_key = state.to_key()

        if self.exploration_strategy == 'ucb':
            # UCB1 exploration strategy: balances exploiting known good actions with exploring less-certain options.
            ucb_values = {}
            for action in ExplorationAction:
                action_str = action.value
                if self.action_counts[state_key][action_str] == 0:
                    # If an action has not been tried, it gets priority.
                    chosen_action = action
                    break
                
                # UCB = average reward + exploration bonus
                average_reward = self.q_table[state_key].get(action_str, 0.0)
                exploration_bonus = math.sqrt(2 * math.log(self.total_steps) / self.action_counts[state_key][action_str])
                ucb_values[action] = average_reward + exploration_bonus
            else: # This else belongs to the for loop, and executes if the loop completes without a break
                chosen_action = max(ucb_values, key=ucb_values.get)

        else: # Default to epsilon-greedy
            # ε-greedy action selection: mostly chooses the best-known action, but with a small probability (epsilon) chooses a random action.
            if random.random() < self.epsilon:
                chosen_action = random.choice(list(ExplorationAction))
            else:
                # Greedy selection based on Q-values
                action_q_values = {}
                for action in ExplorationAction:
                    action_q_values[action] = self.q_table.get(state_key, {}).get(action.value, 0.0)

                best_action = max(action_q_values.keys(), key=lambda k: action_q_values[k])

                # Enhanced early stopping: if STOP_AND_EVALUATE has high Q-value, prefer it
                stop_q = action_q_values.get(ExplorationAction.STOP_AND_EVALUATE, 0.0)
                continue_q = action_q_values.get(ExplorationAction.CONTINUE_EXPLORING, 0.0)

                # If stopping is significantly better than continuing, and we're confident, stop early
                if stop_q > continue_q + 0.5 and stop_q > 0.3:  # Configurable thresholds
                    chosen_action = ExplorationAction.STOP_AND_EVALUATE
                else:
                    chosen_action = best_action
        
        self.action_counts[state_key][chosen_action.value] += 1

        # Log the decision
        if telemetry:
            try:
                from .telemetry_ext import meta_decision
                exploration_state_dict = {
                    "strategy_path": state.strategy_path,
                    "current_depth": state.current_depth,
                    "partial_score": state.partial_score
                }
                meta_decision(telemetry, state_key, chosen_action.value,
                            self.get_q_value(state_key, chosen_action), exploration_state_dict)
            except Exception:
                pass  # Don't break if telemetry fails

        return chosen_action

    def should_stop_early(self, state: ExplorationState, threshold: float = 0.7) -> bool:
        """
        Determine if exploration should stop early based on strategy quality.

        Args:
            state: Current exploration state
            threshold: Minimum score to consider stopping

        Returns:
            True if exploration should stop early
        """
        # Stop if partial score is above threshold
        if state.partial_score >= threshold:
            return True

        # Stop if we've explored deeply enough and score is reasonable
        if state.current_depth >= 3 and state.partial_score >= 0.5:
            return True

        # Stop if we're on a promising path (high score, reasonable depth)
        if state.partial_score >= 0.6 and state.current_depth >= 2:
            return True

        return False

    def get_early_stop_decision(self, state: ExplorationState) -> ExplorationAction:
        """
        Get the best action when considering early stopping.
        """
        if self.should_stop_early(state):
            return ExplorationAction.STOP_AND_EVALUATE

        # Otherwise, use normal Q-learning decision
        return self.decide_exploration(state, [])

    def learn_from_outcome(self, state_key: str, action: ExplorationAction,
                          immediate_reward: float, final_score: float, telemetry=None, is_intermediate: bool = False):
        """Update Q-values based on exploration outcome using Q-learning update rule."""
        old_q = self.q_table[state_key].get(action.value, 0.0)

        # Reward combines immediate efficiency + final strategy quality
        total_reward = immediate_reward + 0.7 * final_score

        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        if is_intermediate:
            # For intermediate steps (reward shaping), we estimate the future reward by looking at the best-known action from the current state.
            # This is a simplified approach; a more advanced method would be to bootstrap from the next state's Q-values.
            next_state_q_values = self.q_table.get(state_key, {}).values()
            max_future_q = max(next_state_q_values) if next_state_q_values else 0.0
            new_q = old_q + self.alpha * (total_reward + self.gamma * max_future_q - old_q)
        else:
            # For terminal states, there is no future reward.
            new_q = old_q + self.alpha * (total_reward - old_q)
            
        self.q_table[state_key][action.value] = new_q

        # Log the learning event
        if telemetry:
            try:
                from .telemetry_ext import meta_learning
                meta_learning(telemetry, state_key, action.value, old_q, new_q, total_reward)
            except Exception:
                pass  # Don't break if telemetry fails

    def save_q_table(self, filepath: str):
        """Persist learned Q-values to JSON file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(dict(self.q_table), f, indent=2)
            print(f"[META] Saved Q-table with {len(self.q_table)} states to {filepath}")
        except Exception as e:
            print(f"[META] Error saving Q-table: {e}")

    def load_q_table(self, filepath: str):
        """Load previously learned Q-values from JSON file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    loaded = json.load(f)
                    self.q_table = defaultdict(dict, loaded)
                print(f"[META] Loaded Q-table with {len(self.q_table)} states from {filepath}")
            except Exception as e:
                print(f"[META] Error loading Q-table: {e}")
                self.q_table = defaultdict(dict)  # Reset to empty

    def save_checkpoint(self, base_path: str, episode: int):
        """Save Q-table checkpoint with episode number."""
        checkpoint_path = f"{base_path}.checkpoint_{episode}"
        self.save_q_table(checkpoint_path)

    def load_latest_checkpoint(self, base_path: str) -> int:
        """Load the most recent checkpoint and return episode number."""
        import glob
        checkpoint_pattern = f"{base_path}.checkpoint_*"
        checkpoints = glob.glob(checkpoint_pattern)

        if not checkpoints:
            return 0

        # Find the checkpoint with highest episode number
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1]))
        episode_num = int(latest_checkpoint.split('_')[-1])

        self.load_q_table(latest_checkpoint)
        return episode_num

    def get_q_value(self, state_key: str, action: ExplorationAction) -> float:
        """Get Q-value for state-action pair."""
        return self.q_table.get(state_key, {}).get(action.value, 0.0)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Q-table for monitoring."""
        total_states = len(self.q_table)
        total_entries = sum(len(actions) for actions in self.q_table.values())
        avg_q_value = 0.0
        if total_entries > 0:
            all_q_values = [q for actions in self.q_table.values() for q in actions.values()]
            avg_q_value = sum(all_q_values) / len(all_q_values)

        return {
            "total_states": total_states,
            "total_q_entries": total_entries,
            "average_q_value": avg_q_value,
            "exploration_rate": self.epsilon
        }