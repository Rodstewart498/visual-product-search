"""Tests for the multi-signal confidence scoring module."""

from visual_search.scoring import compute_confidence, rank_results


class TestComputeConfidence:
    """Tests for the weighted confidence scorer."""

    def test_perfect_match_high_score(self):
        score = compute_confidence(
            orb_distance=10, good_matches=50,
            faiss_distance=0.01, shape_score=95,
            shape_available=True,
        )
        assert score > 80

    def test_poor_match_low_score(self):
        score = compute_confidence(
            orb_distance=200, good_matches=2,
            faiss_distance=0.9, shape_score=10,
            shape_available=True,
        )
        assert score < 40

    def test_shape_increases_score_when_strong(self):
        without_shape = compute_confidence(
            orb_distance=50, good_matches=20,
            faiss_distance=0.3, shape_available=False,
        )
        with_shape = compute_confidence(
            orb_distance=50, good_matches=20,
            faiss_distance=0.3, shape_score=90,
            shape_available=True,
        )
        assert with_shape > without_shape

    def test_more_matches_higher_score(self):
        few = compute_confidence(
            orb_distance=40, good_matches=5,
            faiss_distance=0.2, shape_available=False,
        )
        many = compute_confidence(
            orb_distance=40, good_matches=30,
            faiss_distance=0.2, shape_available=False,
        )
        assert many > few

    def test_lower_orb_distance_higher_score(self):
        far = compute_confidence(
            orb_distance=150, good_matches=20,
            faiss_distance=0.2, shape_available=False,
        )
        close = compute_confidence(
            orb_distance=20, good_matches=20,
            faiss_distance=0.2, shape_available=False,
        )
        assert close > far

    def test_score_always_non_negative(self):
        score = compute_confidence(
            orb_distance=999, good_matches=0,
            faiss_distance=5.0, shape_score=0,
            shape_available=True,
        )
        assert score >= 0

    def test_score_within_range(self):
        score = compute_confidence(
            orb_distance=0, good_matches=100,
            faiss_distance=0, shape_score=100,
            shape_available=True,
        )
        assert 0 <= score <= 100


class TestRankResults:
    """Tests for result ranking."""

    def test_ranks_by_score_descending(self):
        results = [
            {"score": 50, "good_matches": 10, "faiss_distance": 0.1},
            {"score": 80, "good_matches": 20, "faiss_distance": 0.2},
            {"score": 30, "good_matches": 5, "faiss_distance": 0.3},
        ]
        ranked = rank_results(results)
        assert ranked[0]["score"] == 80
        assert ranked[1]["score"] == 50
        assert ranked[2]["score"] == 30

    def test_tiebreak_by_matches(self):
        results = [
            {"score": 50, "good_matches": 5, "faiss_distance": 0.1},
            {"score": 50, "good_matches": 20, "faiss_distance": 0.1},
        ]
        ranked = rank_results(results)
        assert ranked[0]["good_matches"] == 20

    def test_tiebreak_by_faiss_distance(self):
        results = [
            {"score": 50, "good_matches": 10, "faiss_distance": 0.5},
            {"score": 50, "good_matches": 10, "faiss_distance": 0.1},
        ]
        ranked = rank_results(results)
        assert ranked[0]["faiss_distance"] == 0.1

    def test_empty_list(self):
        assert rank_results([]) == []
