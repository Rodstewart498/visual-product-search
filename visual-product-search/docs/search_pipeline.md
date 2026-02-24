# Search Pipeline — Technical Detail

## Signal 1: FAISS HSV Histograms (Color Fingerprint)

### Why HSV over RGB?
HSV separates chromatic content (Hue, Saturation) from brightness (Value). 
This means a red object under bright vs. dim lighting produces similar 
histograms — the Hue and Saturation stay consistent while only Value changes.
We further stabilize this with CLAHE equalization on the V channel.

### Histogram Design
- **Configurable Hue × Saturation bins** (set via `HSV_H_BINS`, `HSV_S_BINS` env vars)
- Hue range: 0-180 (OpenCV convention)
- Saturation range: 0-256
- L2-normalized for consistent FAISS distance calculation
- Small epsilon (1e-8) added to avoid zero-vector issues

### Index Types
- **FlatL2** (< 1,000 images): Exact exhaustive search. Simple, accurate.
- **IVFFlat** (1,000+ images): Inverted file index. Clusters vectors, only 
  searches `nprobe` nearest clusters. Trades ~5% accuracy for 10-50x speed.
  Cluster count: `√N` (e.g., 105 clusters for 11,000 images).

### What It Catches / Misses
- ✅ Overall color distribution — quickly eliminates obviously wrong items
- ❌ Two items with similar color but different form look identical
- ❌ Surface texture, labels, and markings are invisible at this level

---

## Signal 2: ORB Keypoint Matching (Surface Detail)

### Feature Extraction
- ORB (Oriented FAST and Rotated BRIEF): binary descriptor, very fast
- 1,500 keypoints per image with tuned parameters:
  - `edgeThreshold=15` (lower than default 31 — catches more edge features)
  - `fastThreshold=20` (lower than default — more sensitive detection)
- CLAHE preprocessing for consistent feature detection across lighting
- Fallback extraction with even more lenient parameters if primary fails

### Matching Algorithm
1. **kNN match** (k=2) using BFMatcher with Hamming distance
2. **Lowe's ratio test**: Accept match only if `best / second_best < threshold`
   (configurable via `ORB_DISTANCE_RATIO` env var).
   This rejects ambiguous matches where two candidates are close
3. **Consistency bonus**: If good matches have low distance variance,
   the overall score improves (rewards confident, uniform matching)
4. **Single-match fallback**: If only one neighbor found (k=1),
   accept if distance < 100 (typical ORB range is 0-256)

### What It Catches / Misses
- ✅ Labels, text, logos, surface patterns, manufacturing marks
- ✅ Texture details that distinguish similar-looking items
- ❌ Fails on smooth, featureless surfaces (solid color objects)
- ❌ Shape-blind — a round and a square item with the same label match equally

---

## Signal 3: Shape Descriptors (Contour/Silhouette)

### Extraction Pipeline
1. Resize to 400px max (consistent scale)
2. Otsu threshold → binary foreground mask
3. Morphological close (fill gaps) + open (remove noise)
4. Find largest contour (main object)

### Three Feature Groups (48 dimensions total)

**Hu Moments (7 values)** — Shape invariants
- Translation, rotation, and scale invariant
- Log-transformed: `sign(h) × log10(|h|)`
- Normalized to [-1, 1] range

**Edge Direction Histogram (36 values)** — Outline orientation
- Canny edge detection masked to object region
- Sobel gradient angles binned into 36 × 10° buckets
- L2-normalized
- This is the most discriminative signal for distinguishing part types

**Geometric Ratios (5 values)** — Proportions
- Aspect ratio (log-scaled, centered at square)
- Solidity (contour area / convex hull area)
- Extent (contour area / bounding rect area)
- Circularity (4πA/P²)
- Convexity (hull perimeter / contour perimeter)

### Matching
Weighted comparison of three feature groups with configurable weights.

Edge directions typically get the highest weight because they're the most
discriminative for distinguishing items of similar overall shape
but different detailed form (e.g., two rectangular items where one
has rounded corners and the other has angular features).

---

## Combined Scoring

### With Shape Data Available
```
W1 × ORB distance quality    (lower distance = higher score)
W2 × ORB match count         (more matches = higher score)
W3 × FAISS distance          (lower = higher score)
W4 × Shape similarity        (0-100 scale)

Weights configured via SCORE_*_W environment variables (must sum to 1.0)
```

### Without Shape Data (Fallback)
```
W1 × ORB distance quality
W2 × ORB match count
W3 × FAISS distance

Weights configured via SCORE_*_NS_W environment variables (must sum to 1.0)
```

### Result Ranking
Primary: confidence score (descending)
Secondary: good match count (descending)
Tertiary: FAISS distance (ascending)
