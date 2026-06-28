"""WikiText benchmark adapter.

This adapter intentionally keeps `datasets` optional. Importing ComputeOS and
running unit tests should not download benchmark packages or data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from computeos.benchmarks.base import Benchmark, BenchmarkItem


@dataclass(frozen=True)
class WikitextPerplexityBenchmark(Benchmark):
    """Load WikiText samples for future perplexity-oriented experiments."""

    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    split: str = "test"
    text_field: str = "text"
    limit: int | None = 32
    min_chars: int = 20

    def items(self) -> list[BenchmarkItem]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The WikiText benchmark requires the optional `datasets` package. "
                "Install it with `pip install datasets`."
            ) from exc

        dataset = load_dataset(self.dataset_name, self.dataset_config, split=self.split)
        items: list[BenchmarkItem] = []
        for row in dataset:
            text = str(row.get(self.text_field, "")).strip()
            if len(text) < self.min_chars:
                continue
            items.append(
                BenchmarkItem(
                    prompt=text,
                    metadata={
                        "dataset": self.dataset_name,
                        "dataset_config": self.dataset_config,
                        "split": self.split,
                    },
                )
            )
            if self.limit is not None and len(items) >= self.limit:
                break
        return items


def wikitext_from_parameters(
    parameters: dict[str, Any],
    limit: int | None,
) -> WikitextPerplexityBenchmark:
    """Construct a WikiText adapter from generic benchmark config parameters."""

    return WikitextPerplexityBenchmark(
        dataset_name=str(parameters.get("dataset_name", "wikitext")),
        dataset_config=str(parameters.get("dataset_config", "wikitext-2-raw-v1")),
        split=str(parameters.get("split", "test")),
        text_field=str(parameters.get("text_field", "text")),
        limit=limit,
        min_chars=int(parameters.get("min_chars", 20)),
    )
