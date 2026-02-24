"""Tests for shape descriptor extraction and matching."""

import numpy as np
import cv2
import pytest

from visual_search.shape_descriptors import (
    extract_shape_descriptor, match_shape_descriptors, SHAPE_DIM,
)


class TestExtractShapeDescriptor:
    """Tests for shape descriptor extraction."""

    def test_output_shape(self, red_square_image):
        desc = extract_shape_descriptor(red_square_image)
        assert desc.shape == (SHAPE_DIM,)

    def test_output_dtype(self, red_square_image):
        desc = extract_shape_descriptor(red_square_image)
        assert desc.dtype == np.float32

    def test_square_vs_circle_different(self, red_square_image, blue_circle_image):
        desc_sq = extract_shape_descriptor(red_square_image)
        desc_ci = extract_shape_descriptor(blue_circle_image)
        distance = np.linalg.norm(desc_sq - desc_ci)
        assert distance > 0.1, "Square and circle should have different shapes"

    def test_same_shape_different_color(self):
        """Same shape in different colors should produce similar descriptors."""
        red_rect = np.ones((200, 200, 3), dtype=np.uint8) * 255
        red_rect[40:160, 60:140] = [200, 30, 30]

        blue_rect = np.ones((200, 200, 3), dtype=np.uint8) * 255
        blue_rect[40:160, 60:140] = [30, 30, 200]

        desc_red = extract_shape_descriptor(red_rect)
        desc_blue = extract_shape_descriptor(blue_rect)

        # Shape descriptors should be similar since the form is the same
        distance = np.linalg.norm(desc_red - desc_blue)
        assert distance < 1.0, "Same shape in different colors should be similar"

    def test_hu_moments_component(self, red_square_image):
        desc = extract_shape_descriptor(red_square_image)
        hu = desc[:7]
        assert all(-1 <= h <= 1 for h in hu), "Hu moments should be in [-1, 1]"

    def test_geometry_component(self, red_square_image):
        desc = extract_shape_descriptor(red_square_image)
        geometry = desc[43:]
        assert len(geometry) == 5
        # Solidity, extent, circularity, convexity should be in [0, 1]
        for g in geometry[1:]:
            assert 0 <= g <= 1.0

    def test_returns_zeros_on_blank_image(self):
        blank = np.ones((200, 200, 3), dtype=np.uint8) * 255
        desc = extract_shape_descriptor(blank)
        assert np.allclose(desc, 0), "Blank image should produce zero descriptor"

    def test_handles_grayscale_input(self):
        gray = np.ones((200, 200), dtype=np.uint8) * 255
        gray[40:160, 40:160] = 50
        desc = extract_shape_descriptor(gray)
        assert desc.shape == (SHAPE_DIM,)

    def test_no_nan_or_inf(self, noise_image):
        desc = extract_shape_descriptor(noise_image)
        assert not np.any(np.isnan(desc))
        assert not np.any(np.isinf(desc))


class TestMatchShapeDescriptors:
    """Tests for shape descriptor comparison."""

    def test_identical_shapes_high_score(self, red_square_image):
        desc = extract_shape_descriptor(red_square_image)
        score = match_shape_descriptors(desc, desc)
        assert score > 90, "Identical descriptors should score > 90"

    def test_different_shapes_lower_score(self, red_square_image, blue_circle_image):
        desc_sq = extract_shape_descriptor(red_square_image)
        desc_ci = extract_shape_descriptor(blue_circle_image)
        score = match_shape_descriptors(desc_sq, desc_ci)
        self_score = match_shape_descriptors(desc_sq, desc_sq)
        assert score < self_score, "Different shapes should score lower"

    def test_returns_zero_on_none(self):
        assert match_shape_descriptors(None, None) == 0.0

    def test_returns_zero_on_wrong_dimension(self):
        short = np.ones(10, dtype=np.float32)
        full = np.ones(SHAPE_DIM, dtype=np.float32)
        assert match_shape_descriptors(short, full) == 0.0

    def test_score_range(self, red_square_image, green_rectangle_image):
        desc1 = extract_shape_descriptor(red_square_image)
        desc2 = extract_shape_descriptor(green_rectangle_image)
        score = match_shape_descriptors(desc1, desc2)
        assert 0 <= score <= 100
