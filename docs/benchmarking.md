# Benchmarking and Reports

ComputeOS benchmarks produce structured `BenchmarkResult` objects that can be
exported as:

- CSV
- JSON
- Markdown
- LaTeX
- HTML

Use:

```python
from computeos.benchmarks.reporting import rows_from_results, export_benchmark_report

rows = rows_from_results(results)
export_benchmark_report(rows, "outputs/benchmark")
```

Counterfactual Runtime Intelligence also exports policy comparison tables in the
same publication-oriented formats.

## Research Honesty

Benchmark exports report measured fields that exist in telemetry. Missing scores
remain blank or `null`. ComputeOS does not fabricate benchmark accuracy.
