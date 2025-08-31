#!/usr/bin/env python3
"""
demo_meta_control.py

Demonstration script for Phase 1 meta controller functionality.
Shows the Q-learning meta controller in action with a simple example.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from exploration_meta import ExplorationMetaLearner, ExplorationState, ExplorationAction
from strategy_scorer import quick_strategy_score, extract_problem_features

def demo_basic_q_learning():
    """Demonstrate basic Q-learning functionality."""
    print("=== Phase 1 Demo: Basic Q-Learning ===")

    # Create meta learner
    learner = ExplorationMetaLearner(alpha=0.1, gamma=0.95, epsilon=0.0)

    # Create a sample exploration state
    state = ExplorationState(
        strategy_path=["Mathematical Induction", "Prove base case"],
        current_depth=2,
        partial_score=0.6,
        problem_features={"problem_type": "combinatorics", "complexity": "medium"}
    )

    print(f"Initial state: {state.strategy_path}")
    print(f"Current depth: {state.current_depth}")
    print(f"Partial score: {state.partial_score}")

    # Make some decisions and learn
    decisions = []
    for i in range(3):
        decision = learner.decide_exploration(state, [])
        decisions.append(decision)
        print(f"Decision {i+1}: {decision.value}")

        # Simulate learning from the decision
        if decision == ExplorationAction.STOP_AND_EVALUATE:
            learner.learn_from_outcome(state.to_key(), decision, 0.1, 0.8)
        elif decision == ExplorationAction.CONTINUE_EXPLORING:
            learner.learn_from_outcome(state.to_key(), decision, -0.05, 0.6)

    print(f"Decisions made: {[d.value for d in decisions]}")

    # Show learned Q-values
    stats = learner.get_stats()
    print(f"Learner stats: {stats}")

    return learner

def demo_strategy_scoring():
    """Demonstrate lightweight strategy scoring."""
    print("\n=== Phase 1 Demo: Strategy Scoring ===")

    # Sample IMO problem
    problem = """
    A line in the plane is called sunny if it is not parallel to any of the x-axis,
    the y-axis, and the line x+y=0. Let n≥3 be a given integer. Determine all
    nonnegative integers k such that there exist n distinct lines in the plane
    satisfying both the following: for all positive integers a and b with a+b≤n+1,
    the point (a,b) is on at least one of the lines; and exactly k of the lines are sunny.
    """

    # Test different strategies
    strategies = [
        ["Direct counting", "Check all possibilities"],  # Low quality
        ["Mathematical Induction", "Prove for base case n=3", "Assume true for n=k", "Prove for n=k+1"],  # High quality
        ["Case analysis", "Consider different values of k"],  # Medium quality
    ]

    print("Strategy quality assessment:")
    for i, strategy in enumerate(strategies, 1):
        score = quick_strategy_score(strategy, problem)
        print(".3f")

    # Show problem features
    features = extract_problem_features(problem)
    print(f"\nProblem features: {features}")

def demo_integration():
    """Demonstrate how components work together."""
    print("\n=== Phase 1 Demo: Integration ===")

    # Sample problem and strategy
    problem = "Prove that the sum of the first n natural numbers is n(n+1)/2."
    strategy_path = ["Mathematical Induction", "Base case n=1", "Inductive hypothesis"]

    # Extract features
    features = extract_problem_features(problem)

    # Create exploration state
    state = ExplorationState(
        strategy_path=strategy_path,
        current_depth=len(strategy_path),
        partial_score=quick_strategy_score(strategy_path, problem),
        problem_features=features
    )

    # Create meta learner
    learner = ExplorationMetaLearner(alpha=0.2, epsilon=0.1)

    print(f"Strategy: {' -> '.join(strategy_path)}")
    print(f"Problem type: {features['problem_type']}")
    print(f"Strategy score: {state.partial_score:.3f}")

    # Make decision
    decision = learner.decide_exploration(state, [])
    print(f"Meta decision: {decision.value}")

    # Simulate learning
    learner.learn_from_outcome(state.to_key(), decision, 0.05, 0.75)

    print("✅ Phase 1 integration successful!")

def main():
    """Run all Phase 1 demonstrations."""
    print("🚀 Phase 1 Meta Controller Demonstration")
    print("=" * 50)

    try:
        demo_basic_q_learning()
        demo_strategy_scoring()
        demo_integration()

        print("\n" + "=" * 50)
        print("✅ Phase 1 Implementation Complete!")
        print("📋 Summary:")
        print("   • ExplorationMetaLearner class with Q-learning")
        print("   • Lightweight strategy scoring functions")
        print("   • Configuration settings added")
        print("   • Comprehensive test suite")
        print("   • Integration demonstration")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()