# visual-product-search

A multi-signal visual similarity search engine for product images, built with FAISS, OpenCV, and NumPy. Combines color histograms, keypoint matching, and contour-based shape descriptors to find visually similar products across large inventories.

**Tested in production with 11,000+ product images.**

---

## 🎯 Problem

Text-based search fails when you have a physical product and need to find what it is in your inventory. You can see it — but you don't know the part number, the exact name, or which of 11,000 SKUs it matches.

Commercial visual search tools exist but are either too generic (trained on consumer goods, not industrial parts) or too expensive for small/mid-size operations.

## ✅ Solution

A three-signal visual search pipeline that narrows candidates progressively:

```
Query Image
    │
    ▼
┌─────────────────────────┐
│  1. FAISS (HSV Color)   │  Fast approximate nearest-neighbor on color histograms
│     Configurable bins   │  Retrieves top candidates from full inventory
│     ~5ms per query      │
└──────────┬──────────────┘
           │ candidates
           ▼
┌─────────────────────────┐
│  2. ORB (Keypoints)     │  Feature matching with Lowe's ratio test
│     1500 features/image │  Filters to items with strong texture/detail matches
│     ~50ms per candidate │
└──────────┬──────────────┘
           │ refined candidates
           ▼
┌─────────────────────────┐
│  3. Shape Descriptors   │  Contour analysis: Hu moments + edge directions
│     48-dim vectors      │  + geometric ratios (aspect, solidity, circularity)
│     ~10ms per candidate │  Distinguishes items with similar color but different form
└──────────┬──────────────┘
           │
           ▼
    Weighted Confidence Score
    (sorted results with % match)
```

Each signal catches what the others miss — color gets you in the neighborhood, keypoints match surface detail, and shape descriptors distinguish forms that look similar in color but are completely different objects.

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────┐
│                  SearchEngine                          │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │   FAISS  │  │   ORB    │  │  Shape           │    │
│  │  Index   │  │  Matcher │  │  Descriptors     │    │
│  │          │  │          │  │                   │    │
│  │ HSV hist │  │ Keypoint │  │ 7 Hu moments     │    │
│  │ H×S bins │  │ kNN+ratio│  │ 36 edge dirs     │    │
│  │ (config) │  │ test     │  │ 5 geometry feats │    │
│  └──────────┘  └──────────┘  └──────────────────┘    │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Confidence Scorer                               │  │
│  │  Configurable signal weights loaded from env     │  │
│  │  Adapts when shape data is/isn't available       │  │
│  └──────────────────────────────────────────────────┘  │
├────────────────────────────────────────────────────────┤
│  Index Builder                                         │
│  Batch process images → extract features → persist     │
└────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### Three-Signal Matching
- **FAISS HSV histograms** — Color fingerprint using configurable Hue×Saturation bins with CLAHE-equalized Value channel. L2-normalized for consistent distance calculation.
- **ORB keypoint matching** — 1,500 oriented FAST features per image with Lowe's ratio test (configurable threshold). Consistency bonus rewards low-variance match distances.
- **Shape descriptors** — 48-dimensional contour vector: 7 log Hu moments (rotation/scale invariant), 36-bin edge direction histogram, and 5 geometric ratios (aspect ratio, solidity, extent, circularity, convexity).

### Adaptive HSV Pre-filtering
Before expensive ORB matching, candidates are filtered by HSV mean and colorfulness score with progressive tolerance relaxation — starts strict, widens automatically if too few candidates pass.

### Configurable Scoring
All signal weights, normalization constants, and matching thresholds are configurable via environment variables. The code shows the architecture; you tune the parameters for your catalog.

### Scalable Index Building
- **Small inventories** (<1,000 images): `IndexFlatL2` for exact search
- **Large inventories** (1,000+): `IndexIVFFlat` with `nlist = √N` clusters and configurable `nprobe`
- Separate index building for multiple image collections (inventory, competitors, etc.)

### Image Preprocessing Pipeline
- Automatic object centering via contour detection
- CLAHE histogram equalization for lighting normalization
- Adaptive patch extraction for consistent feature regions
- Fallback parameters when primary feature extraction yields no results

---

## 📁 Project Structure

```
visual-product-search/
├── visual_search/
│   ├── __init__.py
│   ├── engine.py             # Main search engine (orchestrates all signals)
│   ├── histograms.py         # HSV histogram extraction + FAISS querying
│   ├── orb_matcher.py        # ORB feature extraction + matching
│   ├── shape_descriptors.py  # Contour-based shape feature extraction
│   ├── preprocessing.py      # Image preprocessing + object centering
│   ├── index_builder.py      # Batch index construction for FAISS + ORB + shape
│   └── scoring.py            # Multi-signal confidence scoring
├── tests/
│   ├── test_histograms.py
│   ├── test_shape_descriptors.py
│   ├── test_scoring.py
│   └── conftest.py
├── docs/
│   └── search_pipeline.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector Search | FAISS (Facebook AI Similarity Search) |
| Feature Detection | OpenCV (ORB, Hu moments, Canny, Sobel) |
| Numerical Computing | NumPy |
| Image I/O | OpenCV imread/cvtColor |
| Index Storage | FAISS binary index + NumPy .npz/.npy files |

---

## 🔧 Setup

### Prerequisites
- Python 3.10+
- System dependencies for OpenCV (usually included with `opencv-python`)

### Installation

```bash
git clone https://github.com/Rodstewart498/visual-product-search.git
cd visual-product-search
pip install -r requirements.txt
```

### Configuration

Signal weights, histogram bins, and matching thresholds are configurable via environment variables:

```bash
# HSV histogram bins (controls color fingerprint resolution)
export HSV_H_BINS=8
export HSV_S_BINS=8

# Scoring weights (with shape data available — must sum to 1.0)
export SCORE_ORB_DIST_W=0.25
export SCORE_ORB_MATCH_W=0.25
export SCORE_FAISS_W=0.25
export SCORE_SHAPE_W=0.25

# ORB matching
export ORB_DISTANCE_RATIO=0.7
```

---

## 📊 Example Usage

### Build an index from a directory of product images

```python
from visual_search.index_builder import build_index

result = build_index(
    image_dir="./product_images/",
    output_dir="./search_index/",
    metadata_path="./metadata.json"
)

print(f"Indexed {result['vectors']} images ({result['dimensions']}d vectors)")
```

### Search for similar products

```python
import cv2
from visual_search.engine import SearchEngine

engine = SearchEngine(index_dir="./search_index/")

query_image = cv2.imread("query_photo.jpg")
query_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

results = engine.search(query_rgb, top_k=10)

for r in results:
    print(f"  {r['filename']} — {r['confidence']:.1f}% "
          f"({r['good_matches']} keypoint matches, "
          f"shape: {r['shape_score']:.1f})")
```

### Extract shape descriptor for a single image

```python
from visual_search.shape_descriptors import extract_shape_descriptor

descriptor = extract_shape_descriptor(image_rgb)
# → 48-dimensional float32 vector:
#   [0:7]   log Hu moments
#   [7:43]  edge direction histogram (36 bins)
#   [43:48] geometric ratios
```

---

## ⚡ Performance

| Operation | Time | Notes |
|-----------|------|-------|
| FAISS search | ~5ms | IVFFlat with configurable nprobe |
| ORB extraction per image | ~15ms | 1,500 features |
| ORB matching per candidate | ~50ms | kNN + ratio test |
| Shape descriptor extraction | ~10ms | Contour + Hu + edge + geometry |
| Full search (11K index) | ~2-3s | FAISS → ORB → ranked results |
| Index build (11K images) | ~8 min | One-time batch operation |

---

## 🧪 How the Three Signals Work Together

| Signal | What It Sees | What It Misses |
|--------|-------------|----------------|
| **FAISS/HSV** | Overall color distribution | Two red items look identical regardless of shape |
| **ORB** | Surface texture, labels, markings | Fails on clean/uniform surfaces |
| **Shape** | Silhouette, proportions, form factor | Color-blind; ignores surface detail |

The weighted combination compensates for each signal's blind spots. In testing, the three-signal approach improved match accuracy by ~40% over FAISS-only search.

---

## 📝 License

© 2025 Rod M. Stewart. All Rights Reserved. This code is provided for portfolio demonstration purposes only. No permission is granted to use, copy, modify, or distribute this software.

---

## 🙋‍♂️ Author

**Rod M. Stewart** — [GitHub](https://github.com/Rodstewart498)

Built to solve real inventory identification problems at scale. If you're managing thousands of physical products and need visual lookup, this architecture delivers.
