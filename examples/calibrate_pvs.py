"""Calibrate PVS weights from real traces and compare perplexity before and after."""

from __future__ import annotations

from computeos.benchmarks.perplexity import PerplexityBenchmark
from computeos.config.schema import ExecutionConfig, ModelConfig, TelemetryConfig
from computeos.execution.engine import InferenceEngine
from computeos.execution.model_loader import load_hf_causal_lm
from computeos.replay.trace_loader import TraceLoader
from computeos.scheduling.pvs import PredictiveValueScheduler, calibrate_weights


def main() -> None:
    model_config = ModelConfig(name="sshleifer/tiny-gpt2")
    loaded = load_hf_causal_lm(model_config)
    execution_config = ExecutionConfig(max_new_tokens=16, seed=42)
    telemetry_config = TelemetryConfig()
    benchmark = PerplexityBenchmark(
        prompts=[f"The future of compute scheduling is step {index}" for index in range(20)],
    )

    pvs = PredictiveValueScheduler()
    engine = InferenceEngine(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        model_name=loaded.name,
        scheduler=pvs,
        execution_config=execution_config,
        telemetry_config=telemetry_config,
    )
    results = benchmark.run(engine)

    loader = TraceLoader()
    traces = [loader.from_telemetry(result.execution.telemetry) for result in results]

    print("Default weights:", pvs.value_weights)
    calibrated = calibrate_weights(traces)
    print("Calibrated weights:", calibrated)

    calibrated_pvs = PredictiveValueScheduler(value_weights=calibrated)
    calibrated_engine = InferenceEngine(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        model_name=loaded.name,
        scheduler=calibrated_pvs,
        execution_config=execution_config,
        telemetry_config=telemetry_config,
    )
    calibrated_results = benchmark.run(calibrated_engine)
    avg_before = sum(result.score or 0 for result in results) / len(results)
    avg_after = sum(result.score or 0 for result in calibrated_results) / len(calibrated_results)
    print(f"Avg perplexity before calibration: {avg_before:.4f}")
    print(f"Avg perplexity after calibration:  {avg_after:.4f}")


if __name__ == "__main__":
    main()
