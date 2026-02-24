"""
visual_search — Multi-signal visual product similarity search.

Combines FAISS (color histograms), ORB (keypoint matching), and
contour-based shape descriptors to find visually similar products
across large inventories.

Modules:
    engine             Main SearchEngine class
    histograms         HSV histogram extraction + FAISS search
    orb_matcher        ORB feature extraction + matching
    shape_descriptors  Contour-based shape features
    preprocessing      Image normalization and centering
    index_builder      Batch index construction
    scoring            Multi-signal confidence scoring
"""

__version__ = "1.0.0"
