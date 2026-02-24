"""
Multi-signal confidence scoring for visual search results.

Combines three independent signals — FAISS color distance, ORB keypoint
matching, and contour shape similarity — into a single confidence
percentage. Each signal compensates for the others' blind spots.

Signal weights are loaded from configuration to allow tuning without
code changes. See DEFAULT_WEIGHTS for the expected structure.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Scoring weights are loaded from environment or config.
# These placeholders are overridden at initialization.
# Tune these values for your specific product catalog and image quality.
DEFAULT_WEIGHTS = {
    "with_shape": {
        "orb_distance": float(os.environ.get("SCORE_ORB_DIST_W", "0.25")),
        "orb_matches": float(os.environ.get("SCORE_ORB_MATCH_W", "0.25")),
        "faiss":       float(os.environ.get("SCORE_FAISS_W", "0.25")),
        "shape":       float(os.environ.get("SCORE_SHAPE_W", "0.25")),
    },
    "without_shape": {
        "orb_distance": float(os.environ.get("SCORE_ORB_DIST_NS_W", "0.34")),
        "orb_matches": float(os.environ.get("SCORE_ORB_MATCH_NS_W", "0.33")),
        "faiss":       float(os.environ.get("SCORE_FAISS_NS_W", "0.33")),
    },
}

# Normalization constants — adjust for your descriptor ranges
ORB_DISTANCE_NORMALIZER = float(os.environ.get("ORB_DIST_NORM", "128.0"))
ORB_DISTANCE_CAP = float(os.environ.get("ORB_DIST_CAP", "2.0"))
MATCH_SCORE_MULTIPLIER = float(os.environ.get("MATCH_SCORE_MULT", "2.0"))


def compute_confidence(orb_distance: float,
                       good_matches: int,
                       faiss_distance: float,
                       shape_score: float = 0.0,
                       shape_available: bool = False,
                       weights: dict = None) -> float:
    """
    Compute a weighted confidence score from multiple visual signals.

    Each signal is normalized to 0-100 before weighting. The weights
    should sum to 1.0 for each mode (with_shape / without_shape).

    Args:
        orb_distance: Mean ORB match distance (lower = better).
        good_matches: Number of ORB matches passing ratio test.
        faiss_distance: L2 distance from FAISS search (lower = better).
        shape_score: Shape descriptor similarity (0-100).
        shape_available: Whether shape descriptors were computed.
        weights: Optional override for scoring weights dict.

    Returns:
        Confidence percentage (0-100).
    """
    weights = weights or DEFAULT_WEIGHTS

    # Normalize ORB distance to 0-100 score
    normalized_distance = min(orb_distance / ORB_DISTANCE_NORMALIZER, ORB_DISTANCE_CAP)
    distance_score = max(0, 100 * (1 - normalized_distance / ORB_DISTANCE_CAP))

    # Match count score: scaled per good match, capped at 100
    match_score = min(100, good_matches * MATCH_SCORE_MULTIPLIER)

    # FAISS distance score (lower is better)
    faiss_score = max(0, 100 - (faiss_distance * 100))

    # Weighted combination
    if shape_available and shape_score > 0:
        w = weights["with_shape"]
        confidence = (
            w["orb_distance"] * distance_score
            + w["orb_matches"] * match_score
            + w["faiss"] * faiss_score
            + w["shape"] * shape_score
        )
    else:
        w = weights["without_shape"]
        confidence = (
            w["orb_distance"] * distance_score
            + w["orb_matches"] * match_score
            + w["faiss"] * faiss_score
        )

    return float(confidence)


def rank_results(results: list) -> list:
    """
    Sort search results by confidence (primary), match count (secondary),
    and FAISS distance (tertiary tiebreaker).

    Args:
        results: List of result dicts with 'score', 'good_matches',
                 and 'faiss_distance' keys.

    Returns:
        Sorted list (highest confidence first).
    """
    return sorted(
        results,
        key=lambda x: (-x['score'], -x['good_matches'], x['faiss_distance'])
    )
