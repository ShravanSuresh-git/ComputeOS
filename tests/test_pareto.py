"""Tests for Pareto frontier analysis."""

from __future__ import annotations

import unittest

from computeos.experiments.pareto import pareto_frontier


class ParetoTests(unittest.TestCase):
    def _point(self, scheduler: str, latency: float, score: float) -> dict[str, object]:
        return {"scheduler": scheduler, "latency_ms": latency, "score": score}

    def test_dominated_point_excluded(self) -> None:
        rows = [
            self._point("A", latency=100.0, score=2.0),
            self._point("B", latency=200.0, score=3.0),
        ]

        names = {point.scheduler for point in pareto_frontier(rows)}

        self.assertIn("A", names)
        self.assertNotIn("B", names)

    def test_tradeoff_points_both_on_frontier(self) -> None:
        rows = [
            self._point("fast", latency=50.0, score=4.0),
            self._point("good", latency=200.0, score=1.5),
        ]

        names = {point.scheduler for point in pareto_frontier(rows)}

        self.assertIn("fast", names)
        self.assertIn("good", names)

    def test_empty_rows_returns_empty(self) -> None:
        self.assertEqual(pareto_frontier([]), [])

    def test_missing_fields_skipped(self) -> None:
        rows = [{"scheduler": "X"}, self._point("Y", 100.0, 2.0)]
        frontier = pareto_frontier(rows)

        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0].scheduler, "Y")


if __name__ == "__main__":
    unittest.main()
