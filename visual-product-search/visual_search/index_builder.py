"""
Batch index construction for FAISS, ORB, and shape descriptors.

Processes a directory of product images and builds the search indexes:
    - FAISS index (.index) — HSV histograms for fast approximate search
    - ORB descriptors (.npz) — keypoint features for detail matching
    - Shape descriptors (.npz) — contour features for form matching
    - Filename mapping (.npy) — maps index positions to image files

Supports both small inventories (FlatL2 exact search) and large
inventories (IVFFlat approximate search with configurable clusters).
"""

import os
import json
import logging
from typing import Optional

import cv2
import faiss
import numpy as np

from .histograms import extract_hsv_histogram, HIST_DIM
from .orb_matcher import extract_orb_features
from .shape_descriptors import extract_shape_descriptor

logger = logging.getLogger(__name__)

# Threshold for switching from exact to approximate FAISS index
IVF_THRESHOLD = 1000


def build_index(image_dir: str,
                output_dir: str,
                metadata_path: Optional[str] = None) -> dict:
    """
    Build all search indexes from a directory of product images.

    Creates three index files in output_dir:
        - faiss_hsv.index — FAISS vector index
        - orb_descriptors.npz — per-image ORB descriptors
        - shape_descriptors.npz — per-image shape descriptors
        - histogram_filenames.npy — ordered filename list

    Args:
        image_dir: Directory containing product images.
        output_dir: Directory to write index files.
        metadata_path: Optional JSON metadata file (must have 'filename'
                       field per entry). If not provided, scans image_dir.

    Returns:
        Dict with 'success', 'vectors', 'dimensions', 'processed' counts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine image list
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        filenames = [e.get('filename') for e in metadata if e.get('filename')]
    else:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        filenames = sorted(
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in extensions
        )

    histograms = []
    orb_descriptors = {}
    shape_descriptors = {}
    valid_filenames = []
    processed = 0
    errors = 0

    logger.info(f"Building index from {len(filenames)} images in {image_dir}")

    for i, filename in enumerate(filenames):
        filepath = os.path.join(image_dir, filename)
        if not os.path.exists(filepath):
            continue

        try:
            image = cv2.imread(filepath)
            if image is None:
                logger.warning(f"Could not read: {filename}")
                errors += 1
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract all three feature types
            histogram = extract_hsv_histogram(image_rgb)
            _, orb_desc = extract_orb_features(image_rgb)
            shape_desc = extract_shape_descriptor(image_rgb)

            histograms.append(histogram)
            valid_filenames.append(filename)

            if orb_desc.size > 0:
                orb_descriptors[filename] = orb_desc

            if np.any(shape_desc != 0):
                shape_descriptors[filename] = shape_desc

            processed += 1

            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i + 1}/{len(filenames)} images")

        except Exception as e:
            logger.warning(f"Failed to process {filename}: {e}")
            errors += 1

    if not histograms:
        return {"success": False, "error": "No valid images processed"}

    # Build FAISS index
    hist_array = np.vstack(histograms).astype(np.float32)
    dim = hist_array.shape[1]

    if len(histograms) >= IVF_THRESHOLD:
        nlist = max(100, int(np.sqrt(len(histograms))))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(hist_array)
        index.add(hist_array)
        logger.info(f"Built IVFFlat index: {nlist} clusters, {dim}d vectors")
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(hist_array)
        logger.info(f"Built FlatL2 index: {dim}d vectors")

    # Save all indexes
    faiss_path = os.path.join(output_dir, "faiss_hsv.index")
    faiss.write_index(index, faiss_path)

    filenames_path = os.path.join(output_dir, "histogram_filenames.npy")
    np.save(filenames_path, np.array(valid_filenames))

    orb_path = os.path.join(output_dir, "orb_descriptors.npz")
    np.savez_compressed(orb_path, **orb_descriptors)

    shape_path = os.path.join(output_dir, "shape_descriptors.npz")
    np.savez_compressed(shape_path, **shape_descriptors)

    logger.info(
        f"Index built: {processed} images, {dim}d vectors, "
        f"{len(orb_descriptors)} ORB, {len(shape_descriptors)} shape, "
        f"{errors} errors"
    )

    return {
        "success": True,
        "processed": processed,
        "vectors": len(histograms),
        "dimensions": dim,
        "orb_count": len(orb_descriptors),
        "shape_count": len(shape_descriptors),
        "errors": errors,
        "index_path": faiss_path,
    }
