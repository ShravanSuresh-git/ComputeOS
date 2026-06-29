"""Run a controlled latency/quality sweep for Predictive Value Scheduling."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import torch
from rich.console import Console
from rich.table import Table
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from computeos.config.schema import ExecutionConfig, TelemetryConfig
from computeos.execution.hf_controlled import HFControlledEngine
from computeos.scheduling.base import Scheduler
from computeos.scheduling.context import SchedulerContext
from computeos.scheduling.decision import SchedulerAction, SchedulerDecision
from computeos.scheduling.pvs import PredictiveValueScheduler, PVSResourceBudgets
from computeos.visualization import plot_pareto_frontier

CANONICAL_CONDITION_ORDER = (
    "baseline",
    "pvs_loose",
    "pvs_medium",
    "pvs_tight",
    "token_cap",
)

BUDGET_PRESETS: dict[str, dict[str, object]] = {
    "baseline": {},
    "pvs_loose": {
        "max_compute_units": 100.0,
        "max_latency_ms": 10_000.0,
        "min_net_value": 0.0,
    },
    "pvs_medium": {
        "max_compute_units": 60.0,
        "max_latency_ms": 10_000.0,
        "min_net_value": 0.0,
    },
    "pvs_tight": {
        "max_compute_units": 30.0,
        "max_latency_ms": 10_000.0,
        "min_net_value": 0.0,
    },
    "token_cap": {
        "max_compute_units": 10_000.0,
        "max_latency_ms": 10_000.0,
        "min_net_value": 0.60,
    },
}


def main() -> None:
    """Run the sweep and write ``outputs/sweep_results.json``."""

    args = _parse_args()
    if args.fast:
        args.n_prompts = 10
        args.max_new_tokens = 10
    pairs = _sample_prompt_continuation_pairs(args.n_prompts)
    model_names = ["distilgpt2", "gpt2-medium"] if args.model == "all" else [args.model]
    primary_results: dict[str, object] | None = None

    for model_index, requested_model in enumerate(model_names):
        if requested_model == "gpt2-medium" and not args.fast:
            Console().print(
                "Warning: gpt2-medium on CPU may take 20+ minutes. "
                "Pass --fast for a quick smoke run."
            )
        try:
            model, tokenizer, model_name = _load_model(requested_model)
        except (RuntimeError, OSError) as exc:
            Console().print(f"Skipping {requested_model}: {exc}")
            continue
        n_layers = _model_layer_count(model)
        scaled_budgets = _scale_budgets_for_model(
            model_name=model_name,
            n_layers=n_layers,
            max_new_tokens=args.max_new_tokens,
        )
        results = run_sweep(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            pairs=pairs,
            max_new_tokens=args.max_new_tokens,
            n_layers=n_layers,
            scaled_budgets=scaled_budgets,
        )
        suffix_path = _outputs_dir() / f"sweep_results_{model_name}.json"
        _write_json(suffix_path, results)
        if model_index == 0:
            primary_results = results
            _write_json(_outputs_dir() / "sweep_results.json", results)
        _print_table(results)

    if primary_results is not None:
        try:
            path = plot_pareto_frontier(primary_results, _outputs_dir() / "pareto_frontier.png")
        except ImportError as exc:
            Console().print(f"Pareto plot skipped: {exc}")
        else:
            Console().print(f"Pareto plot saved to {path}")


def run_sweep(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    model_name: str,
    pairs: list[tuple[str, str]],
    max_new_tokens: int,
    n_layers: int,
    scaled_budgets: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Evaluate baseline and PVS budget variants on a shared prompt set."""

    max_continuation_tokens = 5
    telemetry_config = TelemetryConfig(capture_memory=True)
    conditions: list[tuple[str, Callable[[], Scheduler]]] = [
        (name, lambda preset=name: _make_pvs_scheduler(preset, scaled_budgets))
        for name in CANONICAL_CONDITION_ORDER
    ]
    random.Random(42).shuffle(conditions)

    per_condition: dict[str, list[dict[str, float]]] = {
        name: [] for name in CANONICAL_CONDITION_ORDER
    }
    for condition, scheduler_factory in conditions:
        engine = HFControlledEngine(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            scheduler=scheduler_factory(),
            execution_config=ExecutionConfig(max_new_tokens=max_new_tokens, use_cache=False),
            telemetry_config=telemetry_config,
        )
        engine.warm_up(prompt=pairs[0][0])
        for prompt_index, (prompt, continuation) in enumerate(pairs):
            started_at = perf_counter()
            execution = engine.generate(prompt)
            wall_latency_ms = (perf_counter() - started_at) * 1000.0
            generated_text = tokenizer.decode(execution.output_ids, skip_special_tokens=True)
            if isinstance(generated_text, list):
                generated_text = "".join(generated_text)
            generated_with_prompt = prompt + str(generated_text)
            log_probs = engine.score_continuation(
                generated_with_prompt,
                continuation,
                max_tokens=max_continuation_tokens,
            )
            score = math.exp(-sum(log_probs) / len(log_probs)) if log_probs else float("inf")
            per_condition[condition].append(
                {
                    "prompt_index": float(prompt_index),
                    "latency_ms": wall_latency_ms,
                    "perplexity": float(score or 0.0),
                    "layers_executed": float(len(execution.telemetry.layers)),
                    "early_exits": float(_early_exits_applied(execution.telemetry)),
                    "earliest_exit_layer_index": float(
                        _earliest_exit_layer_index(execution.telemetry)
                    ),
                }
            )

    baseline_latency = mean(row["latency_ms"] for row in per_condition["baseline"])
    baseline_perplexity = mean(row["perplexity"] for row in per_condition["baseline"])
    summaries: dict[str, dict[str, float]] = {}
    for condition in CANONICAL_CONDITION_ORDER:
        rows = per_condition[condition]
        latency_values = [row["latency_ms"] for row in rows]
        perplexity_values = [row["perplexity"] for row in rows]
        mean_latency = mean(latency_values)
        mean_perplexity = mean(perplexity_values)
        summaries[condition] = {
            "mean_latency_ms": mean_latency,
            "std_latency_ms": _std(latency_values),
            "mean_perplexity": mean_perplexity,
            "std_perplexity": _std(perplexity_values),
            "mean_layers_executed": mean(row["layers_executed"] for row in rows),
            "mean_early_exits": mean(row["early_exits"] for row in rows),
            "mean_earliest_exit_layer_index": mean(
                row["earliest_exit_layer_index"] for row in rows
            ),
            "latency_reduction_pct": 100.0 * (baseline_latency - mean_latency) / baseline_latency
            if baseline_latency > 0.0
            else 0.0,
            "perplexity_delta": baseline_perplexity - mean_perplexity,
        }

    return {
        "conditions": summaries,
        "n_prompts": len(pairs),
        "model": model_name,
        "n_layers": n_layers,
        "perplexity_metric": (
            "reference perplexity of continuation given condition-generated text; "
            "perplexity_delta is baseline minus condition"
        ),
        "reference_dataset": "curated_diverse_pairs",
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _make_pvs_scheduler(
    preset: str,
    budgets: dict[str, dict[str, object]] | None = None,
) -> Scheduler:
    """Create a scheduler for a named budget preset."""

    budgets = budgets or BUDGET_PRESETS
    if preset == "baseline":
        return FullExecutionScheduler()
    if preset == "default":
        return PredictiveValueScheduler()
    if preset == "tight":
        preset = "pvs_tight"
    parameters = budgets.get(preset)
    if parameters is None:
        raise ValueError(f"Unknown PVS budget preset: {preset}")
    return PredictiveValueScheduler(
        budgets=PVSResourceBudgets(
            max_latency_ms=float(parameters["max_latency_ms"]),
            max_compute_units=float(parameters["max_compute_units"]),
            min_net_value=float(parameters["min_net_value"]),
        )
    )


def _scale_budgets_for_model(
    model_name: str,
    n_layers: int,
    max_new_tokens: int,
) -> dict[str, dict[str, object]]:
    """Return BUDGET_PRESETS scaled to model depth and generation length."""

    _ = model_name
    total_compute_units = float(max(1, n_layers) * max(1, max_new_tokens))
    return {
        "baseline": {},
        "pvs_loose": {
            "max_compute_units": 0.83 * total_compute_units,
            "max_latency_ms": 10_000.0,
            "min_net_value": 0.0,
        },
        "pvs_medium": {
            "max_compute_units": 0.50 * total_compute_units,
            "max_latency_ms": 10_000.0,
            "min_net_value": 0.0,
        },
        "pvs_tight": {
            "max_compute_units": 0.25 * total_compute_units,
            "max_latency_ms": 10_000.0,
            "min_net_value": 0.0,
        },
        "token_cap": dict(BUDGET_PRESETS["token_cap"]),
    }


def _model_layer_count(model: PreTrainedModel) -> int:
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "h"):
        return len(transformer.h)
    config_layers = getattr(getattr(model, "config", None), "num_hidden_layers", 6)
    return int(config_layers)


def _sample_prompt_continuation_pairs(n_prompts: int) -> list[tuple[str, str]]:
    """Sample prompt/reference-continuation pairs for reference perplexity."""

    return _fallback_pairs(n_prompts)


def _sample_prompts(n_prompts: int) -> list[str]:
    """Compatibility helper for examples that only need prompts."""

    return [prompt for prompt, _continuation in _sample_prompt_continuation_pairs(n_prompts)]


def _append_article_pair(article_lines: list[str], pairs: list[tuple[str, str]]) -> None:
    article = " ".join(" ".join(line.split()) for line in article_lines)
    if len(article) < 320:
        return
    prompt, continuation = _split_reference_text(article)
    if prompt and continuation and _is_reference_pair_candidate(prompt, continuation):
        pairs.append((prompt, continuation))


def _split_reference_text(text: str) -> tuple[str, str]:
    prompt_end = text.rfind(" ", 100, 151)
    if prompt_end < 0:
        return "", ""
    continuation_end = text.rfind(" ", prompt_end + 100, prompt_end + 151)
    if continuation_end < 0:
        return "", ""
    return text[: prompt_end + 1], text[prompt_end + 1 : continuation_end]


def _is_reference_pair_candidate(prompt: str, continuation: str) -> bool:
    text = prompt + continuation
    if any(char.isdigit() for char in text):
        return False
    if any(marker in text for marker in ("@", "(", ")", "[", "]", "=", "–")):
        return False
    punctuation = sum(text.count(mark) for mark in (",", ";", ":", '"', "'"))
    if punctuation > 8:
        return False
    words = text.split()
    if len(words) < 35:
        return False
    short_caps = sum(1 for word in words if len(word) > 1 and word.isupper())
    return short_caps <= 2


class FullExecutionScheduler(Scheduler):
    """Scheduler baseline that never requests adaptive runtime actions."""

    def reset(self) -> None:
        """No state is maintained between prompts."""

    def decide(self, context: SchedulerContext) -> SchedulerDecision:
        """Record the decision point and continue full execution."""

        return SchedulerDecision(
            action=SchedulerAction.RECORD_ONLY,
            layer_name=context.layer_name,
            reason="full execution baseline",
        )


def _fallback_pairs(n_prompts: int) -> list[tuple[str, str]]:
    pairs = [
        (
            "The field crew crossed the dry streambed slowly, because the exposed shale "
            "preserved leaf impressions from a forest that had vanished",
            "long before the valley became grassland, giving the expedition a rare record of "
            "climate change across thousands of seasons.",
        ),
        (
            "When astronomers compare the color of a distant star with its measured "
            "brightness, they can estimate its temperature and",
            "separate nearby dwarf stars from remote giants whose light has traveled for "
            "centuries before reaching the telescope.",
        ),
        (
            "In the tide pool, the small anemones closed whenever a shadow crossed the "
            "water, while limpets continued scraping algae",
            "from the rocks with slow regular movements that made the whole pool seem less "
            "fragile than it first appeared.",
        ),
        (
            "Chemists studying the old pigment found that the blue powder darkened only "
            "when moisture carried salts into the paint",
            "and that careful control of humidity could preserve murals without removing "
            "them from the chapel walls.",
        ),
        (
            "The botanist noted that seedlings near the fallen trunk survived the drought "
            "better than those in open soil because",
            "the decaying wood stored water, sheltered roots, and slowly returned minerals "
            "to the forest floor.",
        ),
        (
            "The compiler team redesigned the parser after discovering that small grammar "
            "ambiguities made error messages confusing",
            "for beginners, even though experienced programmers could usually infer what "
            "the language intended.",
        ),
        (
            "A routing protocol can look simple in a diagram, yet the real network must "
            "cope with dropped packets, asymmetric paths",
            "and sudden congestion that appears only when thousands of machines send data "
            "at the same time.",
        ),
        (
            "The database migration was scheduled for dawn, when traffic was lowest and "
            "engineers could compare old indexes",
            "with the new query plans before customer dashboards began refreshing for the "
            "business day.",
        ),
        (
            "In a distributed system, a timeout is not proof that another server has "
            "failed; it is only evidence that",
            "a message did not return before the local process had to make another decision "
            "under uncertainty.",
        ),
        (
            "The robotics students learned that a path planner which worked perfectly in "
            "simulation still needed margins for",
            "loose cables, uneven floors, imperfect wheels, and people who stepped into the "
            "hallway without warning.",
        ),
        (
            "The treaty negotiations continued through the winter because each delegation "
            "wanted guarantees that the border commission",
            "would include surveyors, interpreters, and merchants familiar with local roads "
            "and winter crossings.",
        ),
        (
            "After the election, the coalition survived by assigning ministries to rival "
            "parties while keeping the budget committee",
            "under a chair trusted by both urban reformers and rural conservatives during "
            "the first session.",
        ),
        (
            "The archive revealed that food shortages shaped the uprising as much as "
            "pamphlets did, since bakers, sailors",
            "and railway workers all described the same morning queues in their letters "
            "home to relatives.",
        ),
        (
            "A new constitution can promise equal rights in elegant language, but courts "
            "and provincial offices determine whether",
            "those promises become ordinary habits in schools, factories, and police stations.",
        ),
        (
            "When the port was blockaded, inland towns learned how dependent their markets "
            "were on imported cloth, salt",
            "and machine parts that had once seemed too mundane to appear in political speeches.",
        ),
        (
            "The novelist opens with a kitchen rather than a battlefield, allowing the "
            "reader to notice how war enters",
            "through ration cards, missing chairs, and the careful silence of older relatives.",
        ),
        (
            "A borrowed word often changes after crossing languages, losing one shade of "
            "meaning while gaining another through",
            "jokes, trade, songs, and the habits of children who pronounce it differently "
            "at school.",
        ),
        (
            "The poet's line break delays the verb just long enough for the image to "
            "hesitate, as if the",
            "speaker were choosing between confession and description while the reader waits.",
        ),
        (
            "In the courtroom speech, repetition does more than decorate the argument; it "
            "builds a rhythm that",
            "lets listeners remember the accusation even after the legal details become tangled.",
        ),
        (
            "The translator kept the proverb literal in the first draft, then realized "
            "that a stranger image",
            "would preserve the humor better than a polished phrase from the target language.",
        ),
        (
            "A grain merchant can profit from rising prices only if warehouses, credit, "
            "and transport contracts allow",
            "the crop to move before rain or insects destroy the stored harvest in the "
            "river warehouses.",
        ),
        (
            "Central banks influence lending not by ordering every bank to change behavior, "
            "but by altering the",
            "price of reserves and the expectations that shape tomorrow's borrowing decisions.",
        ),
        (
            "The factory manager tracked delays back to a single imported gasket, showing "
            "how a cheap part",
            "could halt an expensive assembly line when suppliers carried no spare inventory.",
        ),
        (
            "A market stall owner may understand inflation before economists publish the "
            "figures, because wholesale prices",
            "change the size of each bundle she can afford to place on display before "
            "customers arrive.",
        ),
        (
            "Trade routes rarely follow the shortest line on a map; they bend toward "
            "safe harbors, reliable wells",
            "and towns where contracts can be enforced without sending soldiers beyond "
            "the city gates.",
        ),
        (
            "The river widens below the plateau, dropping silt along the inner banks "
            "where farmers plant melons",
            "and leaving gravel bars on the bends exposed to stronger currents after "
            "spring flooding.",
        ),
        (
            "On the windward side of the island, clouds gather against the ridge and "
            "release steady rain, while",
            "the leeward villages manage orchards with cisterns and careful pruning "
            "through dry months.",
        ),
        (
            "The desert pavement looked empty from the road, but between the stones tiny "
            "plants waited for",
            "brief storms that could turn a shallow wash green within days before the "
            "heat returned.",
        ),
        (
            "A glacier carves the valley slowly, grinding bedrock into flour and leaving "
            "moraines that later",
            "mark where the ice paused during colder decades before retreating up the valley.",
        ),
        (
            "Mangrove roots trap sediment at the edge of the lagoon, creating nurseries "
            "for fish and",
            "protecting the village from waves that arrive during late summer storms and "
            "highest tides.",
        ),
        (
            "The skeptic did not deny that the witness was sincere; she questioned whether "
            "memory alone",
            "could justify certainty after distance, darkness, and fear had shaped the scene.",
        ),
        (
            "A moral rule that cannot survive ordinary exceptions may still teach something, "
            "because the",
            "exceptions reveal which human goods the rule was trying to protect in daily "
            "practice.",
        ),
        (
            "In formal logic, an argument can be valid even when its premises are false, "
            "since validity",
            "concerns the structure that carries truth from assumptions to conclusion "
            "across cases.",
        ),
        (
            "The old debate about free will changes when prediction becomes statistical "
            "rather than absolute, because",
            "probability leaves room for responsibility without making choice mysterious "
            "or random.",
        ),
        (
            "A claim of knowledge requires more than confidence; it also requires reasons "
            "that remain",
            "stable when another person asks how the belief could have been mistaken in "
            "the first place.",
        ),
        (
            "The proof begins by assuming there are only finitely many primes, then "
            "constructs a number",
            "that leaves a remainder when divided by every prime on the list, forcing a "
            "contradiction.",
        ),
        (
            "A triangle drawn on a sphere can have three right angles, reminding students "
            "that familiar",
            "Euclidean rules depend on the surface where geometry is being measured and "
            "drawn.",
        ),
        (
            "The sequence seemed random until the mathematician plotted successive ratios, "
            "where a simple",
            "limit appeared behind the growing list of integers and suggested a hidden "
            "pattern.",
        ),
        (
            "In graph theory, a bridge is an edge whose removal disconnects the network, "
            "making it",
            "important for understanding roads, circuits, and fragile communication "
            "systems under stress.",
        ),
        (
            "A probability model can predict the average number of arrivals while still "
            "failing to",
            "tell the clerk whether the next hour will feel empty or overwhelmed at the "
            "counter.",
        ),
        (
            "The immune cell recognized the infected tissue because small fragments of "
            "viral protein were",
            "displayed on the cell surface like warning flags for nearby immune patrols "
            "to inspect.",
        ),
        (
            "A drug that works well in a dish may fail in the body when enzymes break "
            "it down",
            "before enough of the compound reaches the diseased tissue in a useful dose "
            "for treatment.",
        ),
        (
            "The surgeon traced the nerve carefully, knowing that a cut of only a few "
            "millimeters",
            "could change sensation in the patient's hand for years after the incision "
            "healed.",
        ),
        (
            "Bacteria in the fermenting vat produced acid faster when the room warmed, "
            "so the",
            "cheesemaker adjusted the timing rather than changing the recipe or starter "
            "culture.",
        ),
        (
            "During sleep, the brain does not simply shut down; it cycles through states "
            "that",
            "alter memory, temperature, hormone release, and muscle tone throughout the "
            "night.",
        ),
        (
            "The jazz ensemble left space after the trumpet solo, letting the bassist "
            "answer with",
            "a short phrase that changed the direction of the tune before the drums "
            "returned.",
        ),
        (
            "The apartment block looked severe from the street, but inside the courtyard "
            "children played",
            "beside vines that softened the concrete balconies and shaded the laundry "
            "lines below.",
        ),
        (
            "A public festival can preserve tradition while changing every year, because "
            "costumes, songs",
            "and food stalls respond to new families joining the neighborhood each spring "
            "season.",
        ),
        (
            "The craft guild trained apprentices through repetition, but mastery also "
            "required knowing",
            "which flaw in the material could become part of the final design rather "
            "than waste.",
        ),
        (
            "The choir director asked for less volume and more attention to consonants, "
            "because the",
            "old stone church blurred every word that was sung too forcefully near the "
            "arches.",
        ),
    ]
    if len(pairs) != 50:
        raise RuntimeError("Fallback reference dataset must contain exactly 50 pairs.")
    return pairs[:n_prompts]

def _load_model(
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model, tokenizer, model_name


def _early_exits_applied(telemetry: object) -> int:
    decisions = getattr(telemetry, "scheduler_decisions", [])
    count = 0
    for decision in decisions:
        metadata = getattr(decision, "metadata", {})
        action_result = metadata.get("action_result") if isinstance(metadata, dict) else None
        if (
            getattr(decision, "action", None) == SchedulerAction.EARLY_EXIT
            and isinstance(action_result, dict)
            and action_result.get("applied") is True
        ):
            count += 1
    return count


def _earliest_exit_layer_index(telemetry: object) -> int:
    decisions = getattr(telemetry, "scheduler_decisions", [])
    for decision in decisions:
        metadata = getattr(decision, "metadata", {})
        action_result = metadata.get("action_result") if isinstance(metadata, dict) else None
        if (
            getattr(decision, "action", None) == SchedulerAction.EARLY_EXIT
            and isinstance(action_result, dict)
            and action_result.get("applied") is True
        ):
            layer_name = getattr(decision, "layer_name", None)
            if isinstance(layer_name, str):
                try:
                    return int(layer_name.rsplit(".", maxsplit=1)[-1])
                except ValueError:
                    return -1
    return -1


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _outputs_dir() -> Path:
    path = Path(__file__).resolve().parents[1] / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_table(results: dict[str, object]) -> None:
    conditions = results["conditions"]
    if not isinstance(conditions, dict):
        return
    table = Table(title="ComputeOS PVS Latency/Quality Sweep")
    table.add_column("Condition", no_wrap=True)
    table.add_column("Latency ms", no_wrap=True)
    table.add_column("Ref PPL (given generated text)", no_wrap=True)
    table.add_column("Layers", no_wrap=True)
    table.add_column("Early exits", no_wrap=True)
    table.add_column("Exit layer", no_wrap=True)
    table.add_column("Latency delta", no_wrap=True)
    table.add_column("PPL delta", no_wrap=True)
    for name, raw in conditions.items():
        row = raw if isinstance(raw, dict) else {}
        table.add_row(
            str(name),
            f"{float(row.get('mean_latency_ms', 0.0)):.2f} ± "
            f"{float(row.get('std_latency_ms', 0.0)):.2f}",
            f"{float(row.get('mean_perplexity', 0.0)):.2f} ± "
            f"{float(row.get('std_perplexity', 0.0)):.3f}",
            f"{float(row.get('mean_layers_executed', 0.0)):.2f}",
            f"{float(row.get('mean_early_exits', 0.0)):.2f}",
            f"{float(row.get('mean_earliest_exit_layer_index', -1.0)):.2f}",
            f"{float(row.get('latency_reduction_pct', 0.0)):.1f}%",
            f"{float(row.get('perplexity_delta', 0.0)):.3f}",
        )
    Console().print(table)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-prompts", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument(
        "--model",
        choices=("distilgpt2", "gpt2-medium", "all"),
        default="distilgpt2",
    )
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
