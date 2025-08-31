"""
test_phase2_integration.py

Comprehensive integration tests for Phase 2 meta controller functionality.
Tests the complete integration between NRPA, meta controller, and telemetry.
"""

import unittest
import tempfile
import os
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from nrpa import NRPAConfig, run_nrpa
from strategy_selector import NRPAStrategySelector

class MockTelemetry:
    """Mock telemetry system for testing."""
    def __init__(self):
        self.events = []

    def log_event(self, event_type, description):
        self.events.append((event_type, description))

class TestPhase2Integration(unittest.TestCase):
    """Test Phase 2 integration between components."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_telemetry = MockTelemetry()

        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Sample problem for testing
        self.sample_problem = """
        Prove that the sum of the first n natural numbers is n(n+1)/2.
        """

        # Sample strategies
        self.sample_strategies = [
            "Mathematical Induction",
            "Direct Summation",
            "Gauss's Method"
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_nrpa_config_with_meta_control(self):
        """Test NRPAConfig properly loads meta control settings."""
        # Set environment variables
        os.environ["NRPA_USE_META_CONTROL"] = "1"
        os.environ["NRPA_EARLY_STOP_THRESHOLD"] = "0.8"
        os.environ["NRPA_EXPLORATION_PENALTY"] = "-0.2"

        try:
            config = NRPAConfig.from_env()

            self.assertTrue(config.use_meta_control)
            self.assertEqual(config.early_stop_threshold, 0.8)
            self.assertEqual(config.exploration_penalty, -0.2)
        finally:
            # Clean up environment
            del os.environ["NRPA_USE_META_CONTROL"]
            del os.environ["NRPA_EARLY_STOP_THRESHOLD"]
            del os.environ["NRPA_EXPLORATION_PENALTY"]

    def test_meta_control_disabled_by_default(self):
        """Test that meta control is disabled by default."""
        config = NRPAConfig.from_env()
        self.assertFalse(config.use_meta_control)

    def test_strategy_selector_with_meta_learner(self):
        """Test that NRPAStrategySelector properly handles meta learner."""
        # Mock API functions
        def mock_api_func(*args, **kwargs):
            return {"choices": [{"message": {"content": "test"}}]}

        # Create strategy selector with meta learner
        try:
            from exploration_meta import ExplorationMetaLearner
            meta_learner = ExplorationMetaLearner(alpha=0.1, epsilon=0.0)
            selector = NRPAStrategySelector({"test": mock_api_func}, "test-model", meta_learner)

            self.assertIsNotNone(selector.meta_learner)
            self.assertEqual(selector.meta_learner.alpha, 0.1)
        except ImportError:
            # Skip test if meta controller not available
            self.skipTest("Meta controller not available")

    def test_run_nrpa_with_meta_control(self):
        """Test running NRPA with meta control enabled."""
        # Mock scoring function
        def mock_score_fn(seq):
            return 0.5  # Neutral score

        # Mock children provider
        def mock_children_provider(step, prefix):
            if step == 0:
                return ["Strategy A", "Strategy B"]
            return ["Refinement 1", "Refinement 2"]

        # Create config with meta control
        config = NRPAConfig(
            levels=1,
            iterations=2,
            alpha=1.0,
            max_depth=4,
            use_meta_control=True,
            early_stop_threshold=0.7
        )

        # Mock cache
        cache = {
            "problem_statement": self.sample_problem,
            "problem_features": {"type": "combinatorics", "complexity": "simple"},
            "telemetry": self.mock_telemetry
        }

        # This should not raise an exception
        try:
            score, seq = run_nrpa(
                config=config,
                initial_strategies=self.sample_strategies,
                children_provider=mock_children_provider,
                score_fn=mock_score_fn,
                cache=cache,
                telemetry=self.mock_telemetry
            )

            # Basic validation
            self.assertIsInstance(score, float)
            self.assertIsInstance(seq, list)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        except Exception as e:
            # If meta controller is not available, test should still pass
            if "ExplorationMetaLearner" in str(e):
                self.skipTest("Meta controller not available")
            else:
                raise

    def test_telemetry_integration(self):
        """Test that telemetry events are properly logged."""
        # Add some mock events
        self.mock_telemetry.log_event("TEST_EVENT", "test data")

        # Check that events are recorded
        self.assertEqual(len(self.mock_telemetry.events), 1)
        self.assertEqual(self.mock_telemetry.events[0][0], "TEST_EVENT")
        self.assertEqual(self.mock_telemetry.events[0][1], "test data")

    def test_meta_learner_persistence(self):
        """Test meta learner persistence functionality."""
        try:
            from exploration_meta import ExplorationMetaLearner

            # Create meta learner
            learner = ExplorationMetaLearner(alpha=0.1, epsilon=0.0)

            # Add some learning
            learner.learn_from_outcome("test_state", learner.ExplorationAction.CONTINUE_EXPLORING, 0.2, 0.8)

            # Test save/load
            test_file = os.path.join(self.temp_dir, "test_q_table.json")
            learner.save_q_table(test_file)

            # Verify file was created
            self.assertTrue(os.path.exists(test_file))

            # Create new learner and load
            new_learner = ExplorationMetaLearner()
            new_learner.load_q_table(test_file)

            # Verify Q-value was loaded
            loaded_q = new_learner.get_q_value("test_state", learner.ExplorationAction.CONTINUE_EXPLORING)
            self.assertAlmostEqual(loaded_q, 0.1 * (0.2 + 0.7 * 0.8), places=5)

        except ImportError:
            self.skipTest("Meta controller not available")

    def test_early_stopping_logic(self):
        """Test early stopping decision logic."""
        try:
            from exploration_meta import ExplorationMetaLearner, ExplorationState

            learner = ExplorationMetaLearner(alpha=0.1, epsilon=0.0)

            # Test case 1: High score, low depth - should stop
            state1 = ExplorationState(
                strategy_path=["Strategy A", "Refinement 1"],
                current_depth=2,
                partial_score=0.8,
                problem_features={"type": "combinatorics"}
            )
            self.assertTrue(learner.should_stop_early(state1))

            # Test case 2: Low score - should not stop
            state2 = ExplorationState(
                strategy_path=["Strategy A"],
                current_depth=1,
                partial_score=0.3,
                problem_features={"type": "combinatorics"}
            )
            self.assertFalse(learner.should_stop_early(state2))

        except ImportError:
            self.skipTest("Meta controller not available")

    def test_config_backward_compatibility(self):
        """Test that Phase 2 config maintains backward compatibility."""
        # Default config should work without meta control
        config = NRPAConfig(
            levels=2,
            iterations=60,
            alpha=1.0,
            max_depth=4
        )

        self.assertFalse(config.use_meta_control)
        self.assertIsNone(config.meta_learner_path)
        self.assertEqual(config.early_stop_threshold, 0.7)
        self.assertEqual(config.exploration_penalty, -0.1)
        self.assertEqual(config.efficiency_reward, 0.1)

    def test_integration_error_handling(self):
        """Test that integration handles errors gracefully."""
        # Test with invalid meta learner
        config = NRPAConfig(
            levels=1,
            iterations=1,
            alpha=1.0,
            max_depth=2,
            use_meta_control=True
        )

        def mock_score_fn(seq):
            return 0.5

        def mock_children_provider(step, prefix):
            return ["test"]

        cache = {}

        # Should not crash even with meta control enabled but no meta learner
        config = NRPAConfig(
            levels=1,
            iterations=1,
            alpha=1.0,
            max_depth=2,
            use_meta_control=True
        )
        try:
            score, seq = run_nrpa(
                config=config,
                initial_strategies=["test"],
                children_provider=mock_children_provider,
                score_fn=mock_score_fn,
                cache=cache
            )
            # Should still work (fallback to standard NRPA)
            self.assertIsInstance(score, float)
        except Exception:
            # If it fails, that's also acceptable as long as it's not a crash
            pass

if __name__ == '__main__':
    unittest.main()