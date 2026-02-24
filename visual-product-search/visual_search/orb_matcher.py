"""
ORB (Oriented FAST and Rotated BRIEF) feature extraction and matching.

Provides keypoint-based matching for visual similarity search. ORB
captures local texture and structural features — labels, markings,
surface patterns, and manufacturing details that color histograms miss.

Uses Lowe's ratio test for robust match filtering and includes a
consistency bonus that rewards uniform match distances.
"""

import os
import cv2
import numpy as np
import logging
from typing import Tuple

from .preprocessing import normalize_image, center_object_vertically

logger = logging.getLogger(__name__)

# Default ORB parameters — tuned for product photography
DEFAULT_N_FEATURES = 1500
DEFAULT_DISTANCE_RATIO = float(os.environ.get("ORB_DISTANCE_RATIO", "0.7"))

# Global BFMatcher instance (stateless, safe to reuse)
_bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def extract_orb_features(image_np: np.ndarray,
                         n_features: int = DEFAULT_N_FEATURES
                         ) -> Tuple[list, np.ndarray]:
    """
    Extract ORB keypoints and descriptors from a product image.

    Process:
        1. Normalize image and center the object
        2. Convert to grayscale with CLAHE equalization
        3. Light Gaussian blur to reduce noise
        4. Extract ORB features with optimized parameters
        5. Fallback to more lenient parameters if initial extraction fails

    Args:
        image_np: RGB uint8 image.
        n_features: Maximum number of features to extract.

    Returns:
        Tuple of (keypoints, descriptors). Descriptors is an ndarray
        of shape (N, 32) where N is the number of features found,
        or an empty array if extraction fails.
    """
    try:
        image_np = normalize_image(image_np)
        img_centered = center_object_vertically(image_np)

        # Convert to grayscale
        if len(img_centered.shape) == 3:
            gray = cv2.cvtColor(img_centered, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_centered

        # CLAHE for better feature detection across lighting conditions
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Light blur to reduce noise without losing features
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Primary ORB detector with optimized parameters
        orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=int(os.environ.get("ORB_EDGE_THRESHOLD", "20")),
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=int(os.environ.get("ORB_FAST_THRESHOLD", "20")),
        )

        keypoints, descriptors = orb.detectAndCompute(gray, None)

        # Fallback: more lenient parameters if primary extraction fails
        if descriptors is None:
            logger.warning("Primary ORB extraction failed, trying fallback")
            orb_fallback = cv2.ORB_create(
                nfeatures=n_features + 500,
                scaleFactor=1.1,
                nlevels=10,
                edgeThreshold=10,
                fastThreshold=10,
            )
            keypoints, descriptors = orb_fallback.detectAndCompute(gray, None)

        if descriptors is None:
            descriptors = np.array([])

        logger.debug(f"Extracted {len(descriptors) if descriptors.size > 0 else 0} ORB features")
        return keypoints, descriptors

    except Exception as e:
        logger.error(f"ORB extraction error: {e}")
        return [], np.array([])


def match_orb_descriptors(query_desc: np.ndarray,
                          target_desc: np.ndarray,
                          distance_ratio: float = DEFAULT_DISTANCE_RATIO
                          ) -> Tuple[float, int]:
    """
    Match ORB descriptors using kNN with Lowe's ratio test.

    The ratio test compares the best match distance to the second-best.
    If they're too close, the match is ambiguous and rejected. This
    dramatically reduces false positives.

    Additionally applies a consistency bonus — if the good matches have
    low variance in their distances, the overall score improves. This
    rewards images where multiple features match at similar quality.

    Args:
        query_desc: Query image descriptors (N, 32).
        target_desc: Target image descriptors (M, 32).
        distance_ratio: Lowe's ratio threshold (lower = stricter).

    Returns:
        Tuple of (adjusted_mean_distance, good_match_count).
        Lower distance = better match. Returns (inf, 0) on failure.
    """
    if query_desc.size == 0 or target_desc is None or target_desc.size == 0:
        return float('inf'), 0

    try:
        matches = _bf_matcher.knnMatch(query_desc, target_desc, k=2)

        if not matches:
            return float('inf'), 0

        good_matches = []
        distances = []

        for match in matches:
            if len(match) == 2:
                m, n = match
                # Lowe's ratio test
                if m.distance < distance_ratio * n.distance:
                    good_matches.append(m)
                    distances.append(m.distance)
            elif len(match) == 1:
                # Single match — accept if distance is reasonable
                m = match[0]
                if m.distance < 100:
                    good_matches.append(m)
                    distances.append(m.distance)

        if not good_matches:
            return float('inf'), 0

        mean_distance = np.mean(distances)

        # Consistency bonus: reward uniform match distances
        std_distance = np.std(distances) if len(distances) > 1 else 0
        consistency_bonus = max(0, (50 - std_distance) / 50)
        adjusted_distance = mean_distance * (1 - consistency_bonus * 0.2)

        return adjusted_distance, len(good_matches)

    except Exception as e:
        logger.error(f"ORB matching error: {e}")
        return float('inf'), 0
