"""
strategy_scorer.py

Lightweight strategy evaluation without full solution generation.
Provides fast assessment of strategy quality for meta controller.
"""

from typing import List, Dict, Any
import re

def quick_strategy_score(strategy_path: List[str], problem_statement: str) -> float:
    """
    Fast strategy quality assessment using heuristic analysis.
    Returns 0.0-1.0 score based on:
    - Strategy coherence and completeness
    - Mathematical soundness indicators
    - Problem-statement alignment

    This is much cheaper than generating full Worker sketches.
    """
    if not strategy_path:
        return 0.0

    strategy_text = " ".join(strategy_path).lower()
    problem_text = problem_statement.lower()

    score = 0.0

    # Strategy completeness indicators (0.3 max)
    if len(strategy_path) >= 2:
        score += 0.2  # Has refinement steps
    if any(word in strategy_text for word in ['prove', 'show', 'find', 'determine']):
        score += 0.1  # Has clear objective

    # Mathematical approach indicators (0.4 max)
    math_keywords = ['induction', 'contradiction', 'case', 'bound', 'count', 'construct', 'prove by']
    math_matches = sum(1 for keyword in math_keywords if keyword in strategy_text)
    score += min(0.4, math_matches * 0.1)

    # Problem alignment indicators (0.3 max)
    problem_keywords = extract_problem_keywords(problem_text)
    alignment_score = sum(1 for keyword in problem_keywords if keyword in strategy_text)
    score += min(0.3, alignment_score * 0.1)

    return min(1.0, score)

def extract_problem_features(problem_statement: str) -> Dict[str, Any]:
    """
    Extract features for meta learning context.
    Used by ExplorationState to provide problem-specific context.
    """
    features = {
        "length": len(problem_statement),
        "has_numbers": bool(re.search(r'\d', problem_statement)),
        "has_inequalities": bool(re.search(r'[<>≤≥]', problem_statement)),
        "problem_type": "unknown"
    }

    # Simple problem type classification
    problem_lower = problem_statement.lower()
    if any(word in problem_lower for word in ['combinatorics', 'count', 'number of', 'arrange', 'select', 'choose']):
        features["problem_type"] = "combinatorics"
    elif any(word in problem_lower for word in ['geometry', 'triangle', 'circle', 'line', 'point', 'angle']):
        features["problem_type"] = "geometry"
    elif any(word in problem_lower for word in ['inequality', 'bound', 'maximum', 'minimum', 'prove that']):
        features["problem_type"] = "inequality"
    elif any(word in problem_lower for word in ['function', 'equation', 'solve', 'polynomial']):
        features["problem_type"] = "algebra"

    # Complexity indicators
    features["complexity"] = "simple"
    if len(problem_statement) > 500:
        features["complexity"] = "complex"
    elif len(problem_statement) > 200:
        features["complexity"] = "medium"

    return features

def extract_problem_keywords(problem_statement: str) -> List[str]:
    """
    Extract key mathematical terms from problem statement for alignment scoring.
    """
    # Direct keyword search for common mathematical terms
    math_terms = {
        'integer', 'number', 'function', 'equation', 'inequality', 'prime',
        'triangle', 'circle', 'line', 'point', 'set', 'graph', 'polygon',
        'sequence', 'series', 'sum', 'product', 'limit', 'derivative',
        'integral', 'matrix', 'vector', 'group', 'ring', 'field',
        'combinatorics', 'geometry', 'algebra', 'analysis', 'topology'
    }

    problem_lower = problem_statement.lower()
    found_keywords = []

    for term in math_terms:
        if term in problem_lower:
            found_keywords.append(term)

    return found_keywords

def extract_strategy_features(strategy_path: List[str]) -> Dict[str, Any]:
    """
    Extracts abstract features from a strategy path for generalization.
    This allows the Q-learning agent to recognize similar strategies
    even if they are worded differently.
    """
    features = {
        "has_induction": False,
        "has_contradiction": False,
        "has_casework": False,
        "has_construction": False,
        "has_bounds": False,
        "num_keywords": 0,
    }
    
    strategy_text = " ".join(strategy_path).lower()
    
    if "induction" in strategy_text:
        features["has_induction"] = True
    if "contradiction" in strategy_text:
        features["has_contradiction"] = True
    if "case" in strategy_text:
        features["has_casework"] = True
    if "construct" in strategy_text:
        features["has_construction"] = True
    if "bound" in strategy_text:
        features["has_bounds"] = True
        
    math_keywords = ['prove', 'show', 'find', 'determine', 'induction', 'contradiction', 'case', 'bound', 'count', 'construct']
    features["num_keywords"] = sum(1 for keyword in math_keywords if keyword in strategy_text)
    
    return features

def get_available_refinements(strategy_prefix: List[str]) -> List[str]:
    """
    Get list of possible refinement actions for current strategy prefix.
    This is used by the meta controller to know what actions are available.
    """
    # This is a placeholder - in practice, this would be integrated with
    # the actual refinement generation logic from strategy_selector.py
    base_refinements = [
        "analyze base cases",
        "consider edge cases",
        "prove by contradiction",
        "use mathematical induction",
        "find a constructive proof",
        "establish bounds",
        "count possibilities",
        "show equivalence",
        "derive from definitions",
        "apply known theorems"
    ]

    # Filter based on strategy prefix to avoid redundant actions
    prefix_text = " ".join(strategy_prefix).lower()

    filtered = []
    for refinement in base_refinements:
        # Avoid suggesting induction if already using induction
        if "induction" in prefix_text and "induction" in refinement:
            continue
        # Avoid suggesting contradiction if already using contradiction
        if "contradiction" in prefix_text and "contradiction" in refinement:
            continue
        filtered.append(refinement)

    return filtered

def select_alternative_action(available_actions: List[str], current_strategy: List[str]) -> str:
    """
    Select an alternative action when meta controller suggests switching branches.
    This provides a different refinement path than what NRPA would normally choose.
    """
    if not available_actions:
        return ""

    # Simple heuristic: prefer actions that are different from current strategy
    current_text = " ".join(current_strategy).lower()

    # Score each action by how different it is from current strategy
    action_scores = []
    for action in available_actions:
        score = 0
        action_words = set(action.lower().split())
        current_words = set(current_text.split())

        # Prefer actions with different keywords
        overlap = len(action_words.intersection(current_words))
        score = len(action_words) - overlap  # Higher score for less overlap

        action_scores.append((score, action))

    # Return action with highest difference score
    action_scores.sort(reverse=True)
    return action_scores[0][1] if action_scores else available_actions[0]