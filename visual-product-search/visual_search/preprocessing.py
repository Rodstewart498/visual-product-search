"""
Image preprocessing pipeline for visual search.

Handles normalization, object centering, and adaptive patch extraction
to ensure consistent feature extraction regardless of image quality,
lighting conditions, or framing.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def normalize_image(image_np: np.ndarray) -> np.ndarray:
    """Ensure image is uint8 RGB format."""
    if image_np.dtype != np.uint8:
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
    return image_np


def center_object_vertically(image_np: np.ndarray) -> np.ndarray:
    """
    Detect the main object in the image and center it vertically.

    Uses Otsu thresholding to find the foreground object, then shifts
    the image so the object's centroid is vertically centered. This
    improves feature extraction consistency across images with
    different framing.

    Args:
        image_np: RGB uint8 image.

    Returns:
        Vertically centered image (same dimensions).
    """
    try:
        image_np = normalize_image(image_np)
        h, w = image_np.shape[:2]

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Invert if background dominates
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        # Find the main object's bounding region
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image_np

        main_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(main_contour)

        if moments["m00"] == 0:
            return image_np

        # Calculate vertical centroid offset
        cy = int(moments["m10"] / moments["m00"])
        cx = int(moments["m01"] / moments["m00"])
        shift_y = h // 2 - cx  # shift to center vertically

        # Apply translation
        m = np.float32([[1, 0, 0], [0, 1, shift_y]])
        centered = cv2.warpAffine(image_np, m, (w, h),
                                  borderMode=cv2.BORDER_REFLECT_101)
        return centered

    except Exception as e:
        logger.warning(f"Object centering failed, using original: {e}")
        return image_np


def extract_center_patch(image_np: np.ndarray,
                         center_x: int = None,
                         center_y: int = None,
                         patch_size: int = None) -> np.ndarray:
    """
    Extract a centered patch from the image for feature extraction.

    Uses adaptive patch sizing based on image dimensions, with
    boundary clamping to ensure the patch stays within the image.

    Args:
        image_np: RGB uint8 image.
        center_x: Horizontal center (defaults to image center).
        center_y: Vertical center (defaults to image center).
        patch_size: Desired patch size (defaults to adaptive).

    Returns:
        Cropped image patch.
    """
    h, w = image_np.shape[:2]

    if center_x is None:
        center_x = w // 2
    if center_y is None:
        center_y = h // 2
    if patch_size is None:
        patch_size = min(w, h, 500)

    half = patch_size // 2

    x1 = max(0, center_x - half)
    x2 = min(w, center_x + half)
    y1 = max(0, center_y - half)
    y2 = min(h, center_y + half)

    # Ensure minimum patch size
    if x2 - x1 < patch_size and w >= patch_size:
        if x1 == 0:
            x2 = min(w, patch_size)
        else:
            x1 = max(0, w - patch_size)
            x2 = w

    if y2 - y1 < patch_size and h >= patch_size:
        if y1 == 0:
            y2 = min(h, patch_size)
        else:
            y1 = max(0, h - patch_size)
            y2 = h

    return image_np[y1:y2, x1:x2]


def compute_average_hsv(image_np: np.ndarray):
    """
    Compute average HSV values and colorfulness score for an image.

    Used as a fast pre-filter before expensive feature matching —
    images with very different average color can be eliminated cheaply.

    Args:
        image_np: RGB uint8 image.

    Returns:
        Tuple of (h_mean, s_mean, v_mean, colorfulness_score).
    """
    try:
        image_np = normalize_image(image_np)
        patch = extract_center_patch(image_np, patch_size=400)
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)

        h_mean = float(np.mean(hsv[:, :, 0]))
        s_mean = float(np.mean(hsv[:, :, 1]))
        v_mean = float(np.mean(hsv[:, :, 2]))

        # Colorfulness: ratio of saturation to value
        colorfulness = s_mean / max(v_mean, 1.0)

        return h_mean, s_mean, v_mean, colorfulness

    except Exception as e:
        logger.error(f"HSV computation failed: {e}")
        return 90.0, 50.0, 128.0, 0.5
