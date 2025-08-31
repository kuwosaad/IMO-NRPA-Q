"""
test_meta_control.py

Unit tests for Phase 1 meta controller functionality.
Tests the core Q-learning components and lightweight scoring.
"""

import unittest
import tempfile
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from exploration_meta import ExplorationMetaLearner, ExplorationState, ExplorationAction
from strategy_scorer import quick_strategy_score, extract_problem_features

class TestExplorationMetaLearner(unittest.TestCase):
    """Test the Q-learning meta controller."""

    def setUp(self):
        """Set up test fixtures."""
        self.learner = ExplorationMetaLearner(alpha=0.1, gamma=0.95, epsilon=0.0)  # No exploration for deterministic tests

    def test_q_learning_update(self):
        """Test basic Q-learning update mechanism."""
        state_key = "test_state"
        action = ExplorationAction.STOP_AND_EVALUATE

        # Initial Q-value should be 0
        initial_q = self.learner.get_q_value(state_key, action)
        self.assertEqual(initial_q, 0.0)

        # Learn from outcome
        self.learner.learn_from_outcome(state_key, action, 0.1, 0.8)

        # Q-value should be updated
        updated_q = self.learner.get_q_value(state_key, action)
        expected_q = 0.1 * (0.1 + 0.7 * 0.8)  # alpha * (immediate_reward + 0.7 * final_score)
        self.assertAlmostEqual(updated_q, expected_q, places=5)

    def test_decision_making(self):
        """Test exploration decision making."""
        state = ExplorationState(
            strategy_path=["Induction", "Base Case"],
            current_depth=2,
            partial_score=0.6,
            problem_features={"problem_type": "combinatorics"}
        )

        # With epsilon=0, should make greedy choice
        decision = self.learner.decide_exploration(state, [])
        self.assertIsInstance(decision, ExplorationAction)

    def test_q_table_persistence(self):
        """Test saving and loading Q-table."""
        # Add some learning
        self.learner.learn_from_outcome("state1", ExplorationAction.CONTINUE_EXPLORING, 0.2, 0.7)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name

        try:
            # Save Q-table
            self.learner.save_q_table(temp_path)
            self.assertTrue(os.path.exists(temp_path))

            # Create new learner and load
            new_learner = ExplorationMetaLearner()
            new_learner.load_q_table(temp_path)

            # Should have learned values
            loaded_q = new_learner.get_q_value("state1", ExplorationAction.CONTINUE_EXPLORING)
            self.assertAlmostEqual(loaded_q, 0.1 * (0.2 + 0.7 * 0.7), places=5)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_stats_reporting(self):
        """Test statistics reporting."""
        # Add some data
        self.learner.learn_from_outcome("state1", ExplorationAction.CONTINUE_EXPLORING, 0.1, 0.5)
        self.learner.learn_from_outcome("state2", ExplorationAction.STOP_AND_EVALUATE, 0.2, 0.8)

        stats = self.learner.get_stats()

        self.assertEqual(stats["total_states"], 2)
        self.assertEqual(stats["total_q_entries"], 2)
        self.assertGreater(stats["average_q_value"], 0)
        self.assertEqual(stats["exploration_rate"], 0.0)

class TestStrategyScorer(unittest.TestCase):
    """Test the lightweight strategy scoring."""

    def test_quick_strategy_score(self):
        """Test strategy quality scoring."""
        problem = "Prove that for any positive integer n, the sum of the first n natural numbers is n(n+1)/2."

        # Empty strategy should score 0
        self.assertEqual(quick_strategy_score([], problem), 0.0)

        # Good induction strategy should score well
        induction_strategy = ["Mathematical Induction", "Prove base case n=1", "Assume true for n=k", "Prove for n=k+1"]
        score = quick_strategy_score(induction_strategy, problem)
        self.assertGreater(score, 0.5)  # Should be reasonably high

        # Poor strategy should score low
        bad_strategy = ["Guess randomly", "Try different numbers"]
        bad_score = quick_strategy_score(bad_strategy, problem)
        self.assertLessEqual(bad_score, 0.31)  # Allow small floating point tolerance

    def test_problem_feature_extraction(self):
        """Test problem feature extraction."""
        # Combinatorics problem
        combo_problem = "How many ways are there to arrange 5 distinct objects?"
        features = extract_problem_features(combo_problem)

        self.assertEqual(features["problem_type"], "combinatorics")
        self.assertTrue(features["has_numbers"])
        self.assertIn("complexity", features)

        # Geometry problem
        geom_problem = "Prove that the sum of angles in a triangle is 180 degrees."
        geom_features = extract_problem_features(geom_problem)

        self.assertEqual(geom_features["problem_type"], "geometry")

    def test_keyword_extraction(self):
        """Test keyword extraction from problems."""
        from strategy_scorer import extract_problem_keywords

        problem = "Find all integers n such that n^2 + n + 1 is prime."
        keywords = extract_problem_keywords(problem)

        # Should contain mathematical terms
        self.assertTrue(any(term in keywords for term in ['integer', 'prime']))

if __name__ == '__main__':
    unittest.main()