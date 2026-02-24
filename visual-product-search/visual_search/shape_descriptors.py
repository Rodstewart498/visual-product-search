"""
Contour-based shape descriptor extraction and matching.

Fills the gap where HSV sees color and ORB sees texture, but neither
understands the overall form of an object. Two items can have identical
color and similar texture but completely different silhouettes.

The shape descriptor is a 48-dimensional vector:
    [0:7]   — 7 log Hu moments (rotation/scale/translation invariant)
    [7:43]  — 36-bin edge direction histogram (outline orientation)
    [43:48] — 5 geometric ratios (aspect, solidity, extent, circularity, convexity)
"""

import os
import cv2
import numpy as np
import logging

from .preprocessing import normalize_image

logger = logging.getLogger(__name__)

# Shape matching weights — tune for your product catalog.
# Must sum to 1.0. Edge directions are typically most discriminative.
SHAPE_HU_WEIGHT = float(os.environ.get("SHAPE_HU_W", "0.33"))
SHAPE_EDGE_WEIGHT = float(os.environ.get("SHAPE_EDGE_W", "0.34"))
SHAPE_GEOM_WEIGHT = float(os.environ.get("SHAPE_GEOM_W", "0.33"))

# Descriptor dimensionality: 7 + 36 + 5 = 48
SHAPE_DIM = 48


def extract_shape_descriptor(image_np: np.ndarray) -> np.ndarray:
    """
    Extract a 48-dimensional shape descriptor from a product image.

    Process:
        1. Resize to consistent scale (400px max dimension)
        2. Otsu threshold to isolate foreground object
        3. Morphological cleanup (close gaps, remove noise)
        4. Find largest contour (main object)
        5. Extract three feature groups:
           a. Log Hu moments — rotation/scale invariant shape signature
           b. Edge direction histogram — outline orientation distribution
           c. Geometric ratios — bounding box and contour statistics

    Args:
        image_np: RGB uint8 image.

    Returns:
        48-dimensional float32 descriptor vector.
        Returns zeros if extraction fails.
    """
    try:
        image_np = normalize_image(image_np)

        # Resize to consistent scale
        h, w = image_np.shape[:2]
        target_size = 400
        scale = target_size / max(h, w)
        if scale < 1.0:
            image_np = cv2.resize(image_np, (int(w * scale), int(h * scale)),
                                  interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np.copy()

        h, w = gray.shape[:2]

        # --- Foreground extraction via Otsu threshold ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Invert if Otsu selected the background
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(SHAPE_DIM, dtype=np.float32)

        main_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(main_contour)

        # Skip noise (contour must be at least 1% of image area)
        if contour_area < (h * w * 0.01):
            return np.zeros(SHAPE_DIM, dtype=np.float32)

        # --- Part 1: Hu Moments (7 values) ---
        hu = _extract_hu_moments(main_contour)

        # --- Part 2: Edge Direction Histogram (36 values) ---
        edge_hist = _extract_edge_directions(gray, binary)

        # --- Part 3: Geometric Ratios (5 values) ---
        geometry = _extract_geometry(main_contour, contour_area)

        # Combine all features
        descriptor = np.concatenate([hu, edge_hist, geometry])
        return descriptor.astype(np.float32)

    except Exception as e:
        logger.error(f"Shape descriptor extraction failed: {e}")
        return np.zeros(SHAPE_DIM, dtype=np.float32)


def _extract_hu_moments(contour: np.ndarray) -> np.ndarray:
    """
    Extract 7 log-transformed Hu moments from a contour.

    Hu moments are invariant to translation, rotation, and scale —
    meaning the same object at different positions and sizes produces
    similar values. Log transform improves numerical stability.

    Returns:
        7-element float32 array, normalized to roughly [-1, 1].
    """
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    log_hu = np.array([
        -np.sign(h) * np.log10(max(abs(h), 1e-20))
        for h in hu_moments
    ], dtype=np.float32)

    # Normalize: typical log Hu values range 1-20
    return np.clip(log_hu / 20.0, -1.0, 1.0)


def _extract_edge_directions(gray: np.ndarray,
                              binary_mask: np.ndarray) -> np.ndarray:
    """
    Extract a 36-bin edge direction histogram from the object region.

    Captures the distribution of edge orientations — items with
    predominantly horizontal edges vs. curved edges vs. diagonal
    features produce different histograms.

    Returns:
        36-element float32 array, L2-normalized.
    """
    # Detect edges within the object mask
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=binary_mask)

    # Compute gradient directions
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    edge_mask = edges > 0
    if np.any(edge_mask):
        angles = np.arctan2(gy[edge_mask], gx[edge_mask])
        angles_deg = np.degrees(angles) % 360

        edge_hist, _ = np.histogram(angles_deg, bins=36, range=(0, 360))
        edge_hist = edge_hist.astype(np.float32)

        norm = np.linalg.norm(edge_hist)
        if norm > 0:
            edge_hist = edge_hist / norm
    else:
        edge_hist = np.zeros(36, dtype=np.float32)

    return edge_hist


def _extract_geometry(contour: np.ndarray,
                       contour_area: float) -> np.ndarray:
    """
    Extract 5 geometric ratios from a contour.

    These capture the overall proportions and regularity of the shape:
        - Aspect ratio (wide vs. tall)
        - Solidity (how much of the convex hull is filled)
        - Extent (how much of the bounding box is filled)
        - Circularity (how close to a perfect circle)
        - Convexity (how smooth/convex the outline is)

    Returns:
        5-element float32 array with values in [0, 1] range.
    """
    x, y, bw, bh = cv2.boundingRect(contour)

    # Aspect ratio — log scale centered at 1.0 (square)
    aspect_ratio = float(bw) / max(bh, 1)
    aspect_feature = np.clip(np.log2(max(aspect_ratio, 0.1)), -2, 2) / 2.0

    # Solidity — contour area / convex hull area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(contour_area) / max(hull_area, 1)

    # Extent — contour area / bounding rectangle area
    rect_area = bw * bh
    extent = float(contour_area) / max(rect_area, 1)

    # Circularity — 4π × area / perimeter²
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * contour_area) / max(perimeter ** 2, 1)

    # Convexity — hull perimeter / contour perimeter
    hull_perimeter = cv2.arcLength(hull, True)
    convexity = float(hull_perimeter) / max(perimeter, 1)

    return np.array([
        aspect_feature,
        solidity,
        extent,
        min(circularity, 1.0),
        min(convexity, 1.0),
    ], dtype=np.float32)


def match_shape_descriptors(query_shape: np.ndarray,
                            target_shape: np.ndarray) -> float:
    """
    Compare two shape descriptors and return a similarity score (0-100).

    Uses weighted comparison across the three feature groups.
    Weights are configurable via environment variables:
        - 25% Hu moments (overall contour shape)
        - 40% Edge directions (most discriminative for form)
        - 35% Geometric ratios (proportions and regularity)

    Args:
        query_shape: 48-dim descriptor for query image.
        target_shape: 48-dim descriptor for target image.

    Returns:
        Similarity score from 0 (no match) to 100 (identical shape).
    """
    try:
        if query_shape is None or target_shape is None:
            return 0.0
        if len(query_shape) != SHAPE_DIM or len(target_shape) != SHAPE_DIM:
            return 0.0

        q_hu, q_edge, q_geom = query_shape[:7], query_shape[7:43], query_shape[43:]
        t_hu, t_edge, t_geom = target_shape[:7], target_shape[7:43], target_shape[43:]

        # Hu moment similarity (Euclidean distance)
        hu_dist = np.linalg.norm(q_hu - t_hu)
        hu_score = max(0, 100 * (1.0 - hu_dist / 2.0))

        # Edge direction similarity (cosine similarity)
        q_norm = np.linalg.norm(q_edge)
        t_norm = np.linalg.norm(t_edge)
        if q_norm > 0 and t_norm > 0:
            edge_cosine = np.dot(q_edge, t_edge) / (q_norm * t_norm)
            edge_score = max(0, edge_cosine * 100)
        else:
            edge_score = 0.0

        # Geometry similarity (Euclidean distance)
        geom_dist = np.linalg.norm(q_geom - t_geom)
        geom_score = max(0, 100 * (1.0 - geom_dist / 2.0))

        # Weighted combination
        return float(
            SHAPE_HU_WEIGHT * hu_score
            + SHAPE_EDGE_WEIGHT * edge_score
            + SHAPE_GEOM_WEIGHT * geom_score
        )

    except Exception as e:
        logger.error(f"Shape matching error: {e}")
        return 0.0
