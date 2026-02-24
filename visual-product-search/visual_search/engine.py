"""
Visual product search engine.

Orchestrates the three-signal search pipeline:
    1. FAISS approximate nearest-neighbor on HSV histograms
    2. ORB keypoint matching with Lowe's ratio test
    3. Shape descriptor comparison for contour similarity

Each signal is independent — if one fails (e.g., no ORB features on a
plain surface), the others still contribute to the score.
"""

import os
import logging
from typing import List, Dict, Any, Optional

import cv2
import faiss
import numpy as np

from .histograms import extract_hsv_histogram, search_faiss_index, filter_by_hsv_similarity
from .orb_matcher import extract_orb_features, match_orb_descriptors
from .shape_descriptors import extract_shape_descriptor, match_shape_descriptors
from .preprocessing import compute_average_hsv
from .scoring import compute_confidence, rank_results

logger = logging.getLogger(__name__)

# Lowe's ratio test threshold — tune for your feature space.
# Lower = stricter matching (fewer but higher-quality matches).
# Configure via environment or pass explicitly to search().
DEFAULT_DISTANCE_RATIO = float(os.environ.get("ORB_DISTANCE_RATIO", "0.7"))


class SearchEngine:
    """
    Multi-signal visual product search engine.

    Loads pre-built FAISS, ORB, and shape indexes from disk, then
    accepts query images and returns ranked similarity results.
    """

    def __init__(self, index_dir: str, nprobe: int = 20):
        """
        Load search indexes from disk.

        Args:
            index_dir: Directory containing index files from build_index().
            nprobe: Number of IVF clusters to probe (higher = more accurate,
                    slower). Only applies to IVFFlat indexes.
        """
        self.index_dir = index_dir
        self.nprobe = nprobe

        # Load FAISS index
        faiss_path = os.path.join(index_dir, "faiss_hsv.index")
        self.faiss_index = faiss.read_index(faiss_path)
        if hasattr(self.faiss_index, 'nprobe'):
            self.faiss_index.nprobe = nprobe
        logger.info(
            f"Loaded FAISS index: {self.faiss_index.ntotal} vectors, "
            f"{self.faiss_index.d}d"
        )

        # Load filename mapping
        filenames_path = os.path.join(index_dir, "histogram_filenames.npy")
        self.filenames = list(np.load(filenames_path, allow_pickle=True))

        # Load ORB descriptors
        orb_path = os.path.join(index_dir, "orb_descriptors.npz")
        if os.path.exists(orb_path):
            orb_data = np.load(orb_path, allow_pickle=True)
            self.orb_descriptors = {k: orb_data[k] for k in orb_data.files}
            logger.info(f"Loaded ORB descriptors for {len(self.orb_descriptors)} images")
        else:
            self.orb_descriptors = {}
            logger.warning("No ORB descriptors found")

        # Load shape descriptors
        shape_path = os.path.join(index_dir, "shape_descriptors.npz")
        if os.path.exists(shape_path):
            shape_data = np.load(shape_path, allow_pickle=True)
            self.shape_descriptors = {k: shape_data[k] for k in shape_data.files}
            logger.info(f"Loaded shape descriptors for {len(self.shape_descriptors)} images")
        else:
            self.shape_descriptors = {}
            logger.warning("No shape descriptors found — shape matching disabled")

    def search(self,
               query_image: np.ndarray,
               top_k: int = 20,
               faiss_candidates: int = 200,
               orb_quota: int = 30,
               min_matches: int = 1,
               distance_ratio: float = None) -> List[Dict[str, Any]]:
        """
        Search for visually similar products.

        Pipeline:
            1. Extract HSV histogram → FAISS search → top candidates
            2. Extract ORB features → match against each candidate
            3. Extract shape descriptor → compare contours
            4. Compute weighted confidence → rank results

        Args:
            query_image: RGB uint8 query image.
            top_k: Maximum number of results to return.
            faiss_candidates: Number of FAISS neighbors to retrieve.
            orb_quota: Maximum candidates to run through ORB matching.
            min_matches: Minimum ORB good matches to include a result.
            distance_ratio: Lowe's ratio test threshold for ORB.
                Defaults to DEFAULT_DISTANCE_RATIO from config.

        Returns:
            List of result dicts sorted by confidence, each containing:
                filename, confidence, score, good_matches, orb_distance,
                faiss_distance, shape_score.
        """
        # Step 1: FAISS search on HSV histograms
        distance_ratio = distance_ratio or DEFAULT_DISTANCE_RATIO
        query_histogram = extract_hsv_histogram(query_image)

        try:
            distances, indices = search_faiss_index(
                self.faiss_index, query_histogram,
                k=faiss_candidates, nprobe=self.nprobe
            )
        except ValueError as e:
            logger.error(f"FAISS search failed: {e}")
            return []

        # Step 2: Extract query ORB features
        _, query_orb = extract_orb_features(query_image)
        if query_orb.size == 0:
            logger.warning("No ORB features in query image")
            return []

        # Step 3: Extract query shape descriptor
        query_shape = extract_shape_descriptor(query_image)
        shape_available = bool(self.shape_descriptors) and np.any(query_shape != 0)

        # Step 4: Score each candidate
        results = []
        candidates_processed = 0

        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.filenames):
                continue
            if candidates_processed >= orb_quota * 2:
                break

            filename = self.filenames[idx]
            faiss_dist = float(distances[0][rank])

            # ORB matching
            target_orb = self.orb_descriptors.get(filename)
            if target_orb is None or target_orb.size == 0:
                candidates_processed += 1
                continue

            orb_dist, good_matches = match_orb_descriptors(
                query_orb, target_orb, distance_ratio
            )

            if good_matches < min_matches:
                candidates_processed += 1
                continue

            # Shape matching
            shape_score = 0.0
            if shape_available:
                target_shape = self.shape_descriptors.get(filename)
                if target_shape is not None:
                    shape_score = match_shape_descriptors(query_shape, target_shape)

            # Compute confidence
            confidence = compute_confidence(
                orb_distance=orb_dist,
                good_matches=good_matches,
                faiss_distance=faiss_dist,
                shape_score=shape_score,
                shape_available=shape_available,
            )

            results.append({
                "filename": filename,
                "confidence": round(confidence, 1),
                "score": round(confidence, 2),
                "good_matches": int(good_matches),
                "orb_distance": round(float(orb_dist), 2),
                "faiss_distance": round(faiss_dist, 4),
                "shape_score": round(shape_score, 1),
                "faiss_rank": rank,
            })

            candidates_processed += 1

        # Rank and trim
        results = rank_results(results)[:top_k]

        logger.info(
            f"Search complete: {candidates_processed} candidates → "
            f"{len(results)} results"
        )

        return results
