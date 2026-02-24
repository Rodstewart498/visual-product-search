"""
HSV histogram extraction and FAISS-based similarity search.

Extracts color fingerprints from product images using configurable
Hue×Saturation bins with CLAHE-equalized Value channel. Histograms
are L2-normalized for consistent distance calculation in FAISS.

Bin dimensions are configurable via environment variables (HSV_H_BINS,
HSV_S_BINS). The histogram approach captures overall color distribution
— fast enough for initial candidate retrieval, but needs ORB and shape
signals to distinguish items with similar coloring.
"""

import os
import cv2
import numpy as np
import faiss
import logging
from typing import Optional, Tuple, List

from .preprocessing import normalize_image, center_object_vertically, extract_center_patch

logger = logging.getLogger(__name__)

# HSV histogram configuration
# Bin counts control the tradeoff between discriminative power and index size.
# Higher values = more precise color matching but larger index footprint.
# Configure via environment or override at initialization.
H_BINS = int(os.environ.get("HSV_H_BINS", "8"))
S_BINS = int(os.environ.get("HSV_S_BINS", "8"))
HIST_DIM = H_BINS * S_BINS


def extract_hsv_histogram(image_np: np.ndarray,
                          center_x: int = None,
                          center_y: int = None) -> np.ndarray:
    """
    Extract an L2-normalized HSV histogram from a product image.

    Process:
        1. Normalize and center the object in frame
        2. Extract an adaptive center patch
        3. Convert to HSV and apply CLAHE to V channel
        4. Compute H×S histogram with configured bin counts
        5. L2-normalize + epsilon to avoid zero vectors

    Args:
        image_np: RGB uint8 image.
        center_x: Optional horizontal center for patch extraction.
        center_y: Optional vertical center for patch extraction.

    Returns:
        Float32 histogram vector with H_BINS * S_BINS dimensions.
    """
    try:
        image_np = normalize_image(image_np)
        image_centered = center_object_vertically(image_np)
        patch = extract_center_patch(image_centered, center_x, center_y)

        # Convert to HSV
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)

        # CLAHE equalization on V channel for lighting normalization
        h_ch, s_ch, v_ch = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_eq = clahe.apply(v_ch)
        hsv = cv2.merge((h_ch, s_ch, v_eq))

        # Compute 2D histogram: Hue (0-180) × Saturation (0-256)
        hist = cv2.calcHist([hsv], [0, 1], None,
                            [H_BINS, S_BINS], [0, 180, 0, 256])

        # Flatten and L2-normalize
        hist_flat = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(hist_flat)
        if norm > 0:
            hist_flat = hist_flat / norm

        # Small epsilon to avoid exact-zero entries
        hist_flat = hist_flat + 1e-8

        return hist_flat

    except Exception as e:
        logger.error(f"HSV histogram extraction failed: {e}")
        return np.ones(HIST_DIM, dtype=np.float32) / HIST_DIM


def search_faiss_index(index: faiss.Index,
                       query_histogram: np.ndarray,
                       k: int = 200,
                       nprobe: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search a FAISS index for nearest neighbors to a query histogram.

    Args:
        index: Loaded FAISS index.
        query_histogram: 256-dim float32 query vector.
        k: Number of neighbors to retrieve.
        nprobe: Number of cluster probes (for IVF indexes).

    Returns:
        Tuple of (distances, indices) arrays, each shape (1, k).

    Raises:
        ValueError: If query dimensions don't match index.
    """
    query = query_histogram.astype(np.float32)

    # L2-normalize the query to match indexed vectors
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm

    query = query.reshape(1, -1)

    if query.shape[1] != index.d:
        raise ValueError(
            f"Query dimension {query.shape[1]} doesn't match "
            f"index dimension {index.d}"
        )

    # Set nprobe for IVF indexes
    if hasattr(index, 'nprobe'):
        index.nprobe = nprobe

    k = min(k, index.ntotal)
    distances, indices = index.search(query, k)

    return distances, indices


def filter_by_hsv_similarity(query_hsv: tuple,
                             query_colorfulness: float,
                             candidates: list,
                             hsv_tolerance: tuple = None,
                             colorfulness_tolerance: float = None) -> set:
    """
    Pre-filter candidates by average HSV similarity.

    This is a cheap O(N) filter applied before expensive ORB matching.
    Eliminates candidates whose overall color is too different from
    the query.

    Args:
        query_hsv: Tuple of (h_mean, s_mean, v_mean) for query.
        query_colorfulness: Colorfulness score for query.
        candidates: List of metadata entries with 'avg_hsv' fields.
        hsv_tolerance: Max allowed (H, S, V) deviation.
        colorfulness_tolerance: Max allowed colorfulness deviation.

    Returns:
        Set of indices into candidates that pass the filter.
    """
    h_q, s_q, v_q = query_hsv
    # Defaults loaded from env or use generic starting points
    if hsv_tolerance is None:
        hsv_tolerance = (
            int(os.environ.get("HSV_H_TOL", "30")),
            int(os.environ.get("HSV_S_TOL", "50")),
            int(os.environ.get("HSV_V_TOL", "50")),
        )
    if colorfulness_tolerance is None:
        colorfulness_tolerance = float(os.environ.get("COLOR_TOL", "0.3"))
    h_tol, s_tol, v_tol = hsv_tolerance
    valid = set()

    for i, entry in enumerate(candidates):
        h_t = entry.get('h_mean', 90)
        s_t = entry.get('s_mean', 50)
        v_t = entry.get('v_mean', 128)
        c_t = entry.get('colorfulness', 0.5)

        # Hue wraps at 180 — use circular distance
        h_diff = min(abs(h_q - h_t), 180 - abs(h_q - h_t))

        if (h_diff <= h_tol
                and abs(s_q - s_t) <= s_tol
                and abs(v_q - v_t) <= v_tol
                and abs(query_colorfulness - c_t) <= colorfulness_tolerance):
            valid.add(i)

    return valid
