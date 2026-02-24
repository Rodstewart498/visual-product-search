"""Tests for HSV histogram extraction and FAISS search."""

import numpy as np
import faiss
import pytest

from visual_search.histograms import (
    extract_hsv_histogram, HIST_DIM, search_faiss_index,
    filter_by_hsv_similarity,
)


class TestExtractHsvHistogram:
    """Tests for HSV histogram extraction."""

    def test_output_shape(self, red_square_image):
        hist = extract_hsv_histogram(red_square_image)
        assert hist.shape == (HIST_DIM,)

    def test_output_dtype(self, red_square_image):
        hist = extract_hsv_histogram(red_square_image)
        assert hist.dtype == np.float32

    def test_l2_normalized(self, red_square_image):
        hist = extract_hsv_histogram(red_square_image)
        # Should be approximately unit length (epsilon adds a tiny amount)
        norm = np.linalg.norm(hist)
        assert 0.99 < norm < 1.05

    def test_different_images_different_histograms(self, red_square_image,
                                                     blue_circle_image):
        hist_red = extract_hsv_histogram(red_square_image)
        hist_blue = extract_hsv_histogram(blue_circle_image)
        distance = np.linalg.norm(hist_red - hist_blue)
        assert distance > 0.1, "Distinct colors should produce different histograms"

    def test_similar_images_similar_histograms(self, red_square_image):
        hist1 = extract_hsv_histogram(red_square_image)
        # Slightly modified version of the same image
        modified = red_square_image.copy()
        modified = np.clip(modified.astype(int) + 5, 0, 255).astype(np.uint8)
        hist2 = extract_hsv_histogram(modified)
        distance = np.linalg.norm(hist1 - hist2)
        assert distance < 0.5, "Similar images should produce similar histograms"

    def test_no_nan_or_inf(self, noise_image):
        hist = extract_hsv_histogram(noise_image)
        assert not np.any(np.isnan(hist))
        assert not np.any(np.isinf(hist))

    def test_graceful_on_small_image(self):
        tiny = np.ones((10, 10, 3), dtype=np.uint8) * 128
        hist = extract_hsv_histogram(tiny)
        assert hist.shape == (HIST_DIM,)

    def test_custom_center(self, red_square_image):
        hist = extract_hsv_histogram(red_square_image, center_x=50, center_y=50)
        assert hist.shape == (HIST_DIM,)


class TestSearchFaissIndex:
    """Tests for FAISS search functionality."""

    def test_basic_search(self, red_square_image, blue_circle_image):
        # Build a small index with two images
        hist1 = extract_hsv_histogram(red_square_image)
        hist2 = extract_hsv_histogram(blue_circle_image)

        data = np.vstack([hist1, hist2]).astype(np.float32)
        index = faiss.IndexFlatL2(HIST_DIM)
        index.add(data)

        # Search for red — should rank red first
        distances, indices = search_faiss_index(index, hist1, k=2)
        assert indices[0][0] == 0  # Self-match should be nearest

    def test_dimension_mismatch_raises(self):
        index = faiss.IndexFlatL2(HIST_DIM)
        index.add(np.ones((1, HIST_DIM), dtype=np.float32))

        wrong_dim = np.ones(128, dtype=np.float32)
        with pytest.raises(ValueError, match="dimension"):
            search_faiss_index(index, wrong_dim)


class TestHsvFiltering:
    """Tests for HSV pre-filtering."""

    def test_filters_by_hue(self):
        candidates = [
            {"h_mean": 10, "s_mean": 100, "v_mean": 128, "colorfulness": 0.5},
            {"h_mean": 90, "s_mean": 100, "v_mean": 128, "colorfulness": 0.5},
        ]
        valid = filter_by_hsv_similarity(
            query_hsv=(10, 100, 128), query_colorfulness=0.5,
            candidates=candidates, hsv_tolerance=(20, 60, 60)
        )
        assert 0 in valid
        assert 1 not in valid

    def test_relaxed_tolerance_accepts_more(self):
        candidates = [
            {"h_mean": 10, "s_mean": 100, "v_mean": 128, "colorfulness": 0.5},
            {"h_mean": 50, "s_mean": 100, "v_mean": 128, "colorfulness": 0.5},
        ]
        strict = filter_by_hsv_similarity(
            (10, 100, 128), 0.5, candidates, hsv_tolerance=(10, 30, 30)
        )
        relaxed = filter_by_hsv_similarity(
            (10, 100, 128), 0.5, candidates, hsv_tolerance=(50, 80, 80)
        )
        assert len(relaxed) >= len(strict)
