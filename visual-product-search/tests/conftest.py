"""Shared test fixtures for visual search tests."""

import numpy as np
import cv2
import pytest


@pytest.fixture
def red_square_image():
    """Generate a 200x200 red square on white background."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    img[40:160, 40:160] = [200, 30, 30]  # Red square
    return img


@pytest.fixture
def blue_circle_image():
    """Generate a 200x200 blue circle on white background."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.circle(img, (100, 100), 60, (30, 30, 200), -1)
    return img


@pytest.fixture
def green_rectangle_image():
    """Generate a 200x200 green rectangle on white background."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    img[30:170, 60:140] = [30, 180, 30]  # Tall green rectangle
    return img


@pytest.fixture
def textured_image():
    """Generate a 200x200 image with texture patterns (good for ORB)."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 200
    # Add checkerboard pattern for strong ORB features
    for y in range(0, 200, 20):
        for x in range(0, 200, 20):
            if (x // 20 + y // 20) % 2 == 0:
                img[y:y+20, x:x+20] = [50, 50, 50]
    return img


@pytest.fixture
def noise_image():
    """Generate a 200x200 random noise image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 255, (200, 200, 3), dtype=np.uint8)
