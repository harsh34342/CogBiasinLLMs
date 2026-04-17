"""
run_all_experiments.py
----------------------
Crash-safe batch runner for the focused two-question study.

Runs all combinations sequentially for one model:
- Biases: anchoring, framing
- Methods: baseline, single_selfhelp, multi_selfhelp, no_debias_agent

Usage:
    python run_all_experiments.py
"""

import csv
import logging
import os
import time
from typing import Dict, List, Tuple

import config
from anchoring import AnchoringExperiment
from decision_agent import DecisionAgent
from debiasing_agent import DebiasingAgent
from evaluator_agent import EvaluatorAgent
from evaluate import save_per_student_csv, save_results_csv
from framing import FramingExperiment
from student_profiles import generate_sequential_student_set, generate_student_profiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all")

MODELS = ["mistralai/Mistral-7B-Instruct-v0.2"]
BIASES = ["anchoring", "framing"]
METHODS = ["baseline", "single_selfhelp", "multi_selfhelp", "no_debias_agent"]

N_STUDENTS = 6
N_PERMUTATIONS = 2
SEED = 42
TEMPERATURE = 0.0

OUTPUT_DIR = os.path.join("results", "demo")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary.csv")
TRADEOFF_CSV = os.path.join(OUTPUT_DIR, "method_tradeoff.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run focused demo experiments with configurable model/output settings"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override demo model name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: results/demo)",
    )
    parser.add_argument(
        "--n-students",
        type=int,
        default=None,
        help="Override number of students for demo run",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=None,
        help="Override number of anchoring permutations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override generation temperature",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing summary/tradeoff files in output-dir before run",
    )
    parser.add_argument(
        "--biases",
        type=str,
        default=None,
        help="Comma-separated subset from: anchoring,framing",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated subset from: baseline,single_selfhelp,multi_selfhelp,no_debias_agent",
    )
    return parser.parse_args()


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_completed_keys() -> set:
    """Read existing summary rows so the runner can resume from the last step."""
    if not os.path.exists(SUMMARY_CSV):
        return set()

    done = set()
    with open(SUMMARY_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["model"], row["bias"], row["method"]))
    return done


def _append_summary_row(row: Dict) -> None:
    """Append a single row to summary.csv, creating headers if needed."""
    write_header = not os.path.exists(SUMMARY_CSV)
    fieldnames = [
        "model",
        "bias",
        "method",
        "metric_name",
        "metric_value",
        "tokens_used",
        "num_calls",
        "runtime_seconds",
    ]
    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _reset_agent_stats(*agents) -> None:
    for agent in agents:
        if agent and hasattr(agent, "reset_usage_stats"):
            agent.reset_usage_stats()


def _collect_total_usage(*agents) -> Tuple[int, int]:
    total_tokens = 0
    total_calls = 0
    for agent in agents:
        if not agent or not hasattr(agent, "get_usage_stats"):
            continue
        stats = agent.get_usage_stats()
        total_tokens += stats["input_tokens"] + stats["output_tokens"]
        total_calls += stats["num_calls"]
    return total_tokens, total_calls


def _build_agents(model_name: str):
    return (
        DecisionAgent(
            model_name=model_name,
            device=config.DEVICE,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            hf_token=config.HF_TOKEN,
        ),
        DebiasingAgent(
            model_name=model_name,
            device=config.DEVICE,
            temperature=TEMPERATURE,
            hf_token=config.HF_TOKEN,
        ),
        EvaluatorAgent(
            model_name=model_name,
            device=config.DEVICE,
            temperature=TEMPERATURE,
            hf_token=config.HF_TOKEN,
        ),
    )


def _run_anchoring_method(
    model_name: str,
    method: str,
    profiles,
    decision_agent,
    debiasing_agent,
    evaluator_agent,
) -> Dict:
    experiment = AnchoringExperiment(decision_agent, debiasing_agent, evaluator_agent)

    _reset_agent_stats(decision_agent, debiasing_agent, evaluator_agent)
    start = time.perf_counter()
    result = experiment.run(
        profiles=profiles,
        n_permutations=N_PERMUTATIONS,
        seed=SEED,
        mitigation=method,
    )
    wall_runtime = time.perf_counter() - start
    tokens_used, num_calls = _collect_total_usage(
        decision_agent, debiasing_agent, evaluator_agent
    )

    save_per_student_csv(
        result,
        os.path.join(OUTPUT_DIR, f"anchoring_{model_name.replace('/', '_')}_{method}_per_student.csv"),
    )

    return {
        "model": model_name,
        "bias": "anchoring",
        "method": method,
        "metric_name": "avg_confidence_d",
        "metric_value": round(result["avg_confidence_d"], 6),
        "tokens_used": tokens_used,
        "num_calls": num_calls,
        "runtime_seconds": round(wall_runtime, 4),
    }


def _run_framing_method(
    model_name: str,
    method: str,
    profiles,
    decision_agent,
    debiasing_agent,
    evaluator_agent,
) -> Dict:
    experiment = FramingExperiment(decision_agent, debiasing_agent, evaluator_agent)

    _reset_agent_stats(decision_agent, debiasing_agent, evaluator_agent)
    start = time.perf_counter()
    result = experiment.run(
        profiles=profiles,
        mitigation=method,
    )
    wall_runtime = time.perf_counter() - start
    tokens_used, num_calls = _collect_total_usage(
        decision_agent, debiasing_agent, evaluator_agent
    )

    save_per_student_csv(
        result,
        os.path.join(OUTPUT_DIR, f"framing_{model_name.replace('/', '_')}_{method}_per_student.csv"),
    )

    return {
        "model": model_name,
        "bias": "framing",
        "method": method,
        "metric_name": "delta",
        "metric_value": round(result["delta"], 6),
        "tokens_used": tokens_used,
        "num_calls": num_calls,
        "runtime_seconds": round(wall_runtime, 4),
    }


def _compute_tradeoff_rows(summary_rows: List[Dict]) -> List[Dict]:
    """
    Compute method-level bias reduction and cost increase percentages vs baseline.
    Anchoring bias score is (1 - avg_confidence_d). Framing bias score is abs(delta).
    """
    grouped: Dict[str, Dict[str, Dict]] = {"anchoring": {}, "framing": {}}
    for row in summary_rows:
        grouped[row["bias"]][row["method"]] = row

    output = []
    for bias in BIASES:
        baseline = grouped[bias].get("baseline")
        if not baseline:
            continue

        baseline_tokens = float(baseline["tokens_used"])
        baseline_metric = float(baseline["metric_value"])
        baseline_bias_score = (
            1.0 - baseline_metric if bias == "anchoring" else abs(baseline_metric)
        )

        for method in METHODS:
            row = grouped[bias].get(method)
            if not row:
                continue
            metric = float(row["metric_value"])
            method_bias_score = 1.0 - metric if bias == "anchoring" else abs(metric)
            denom = baseline_bias_score if baseline_bias_score > 0 else 1e-9
            bias_reduction_pct = ((baseline_bias_score - method_bias_score) / denom) * 100.0

            tokens = float(row["tokens_used"])
            cost_denom = baseline_tokens if baseline_tokens > 0 else 1e-9
            cost_increase_pct = ((tokens - baseline_tokens) / cost_denom) * 100.0

            output.append(
                {
                    "bias": bias,
                    "method": method,
                    "baseline_bias_score": round(baseline_bias_score, 6),
                    "method_bias_score": round(method_bias_score, 6),
                    "bias_reduction_pct": round(bias_reduction_pct, 4),
                    "baseline_tokens": int(baseline_tokens),
                    "method_tokens": int(tokens),
                    "cost_increase_pct": round(cost_increase_pct, 4),
                }
            )

    return output


def _load_summary_rows() -> List[Dict]:
    if not os.path.exists(SUMMARY_CSV):
        return []
    with open(SUMMARY_CSV, "r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    global OUTPUT_DIR, SUMMARY_CSV, TRADEOFF_CSV
    global MODELS, N_STUDENTS, N_PERMUTATIONS, SEED, TEMPERATURE

    if args.model:
        MODELS = [args.model]
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    if args.n_students is not None:
        N_STUDENTS = args.n_students
    if args.n_permutations is not None:
        N_PERMUTATIONS = args.n_permutations
    if args.seed is not None:
        SEED = args.seed
    if args.temperature is not None:
        TEMPERATURE = args.temperature
    if args.biases:
        BIASES = [b.strip() for b in args.biases.split(",") if b.strip()]
    if args.methods:
        METHODS = [m.strip() for m in args.methods.split(",") if m.strip()]

    SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary.csv")
    TRADEOFF_CSV = os.path.join(OUTPUT_DIR, "method_tradeoff.csv")

    if args.clear_output:
        for p in (SUMMARY_CSV, TRADEOFF_CSV):
            if os.path.exists(p):
                os.remove(p)

    ensure_output_dir()
    completed = _load_completed_keys()

    logger.info("Starting focused two-question overnight run")
    logger.info(
        "Fixed constants: students=%s permutations=%s seed=%s temperature=%s",
        N_STUDENTS,
        N_PERMUTATIONS,
        SEED,
        TEMPERATURE,
    )

    for model_name in MODELS:
        logger.info("Building agents for model: %s", model_name)
        decision_agent, debiasing_agent, evaluator_agent = _build_agents(model_name)

        anchoring_profiles = generate_sequential_student_set(N_STUDENTS, seed=SEED)
        framing_profiles = generate_student_profiles(N_STUDENTS, seed=SEED)

        for bias in BIASES:
            for method in METHODS:
                key = (model_name, bias, method)
                if key in completed:
                    logger.info("Skipping completed run: %s | %s | %s", *key)
                    continue

                logger.info("Running: model=%s bias=%s method=%s", model_name, bias, method)
                try:
                    if bias == "anchoring":
                        row = _run_anchoring_method(
                            model_name,
                            method,
                            anchoring_profiles,
                            decision_agent,
                            debiasing_agent,
                            evaluator_agent,
                        )
                    else:
                        row = _run_framing_method(
                            model_name,
                            method,
                            framing_profiles,
                            decision_agent,
                            debiasing_agent,
                            evaluator_agent,
                        )

                    _append_summary_row(row)
                    completed.add(key)
                    logger.info("Completed and checkpointed: %s | %s | %s", *key)
                except Exception as exc:
                    logger.exception(
                        "Run failed for model=%s bias=%s method=%s: %s",
                        model_name,
                        bias,
                        method,
                        exc,
                    )
                    raise

    summary_rows = _load_summary_rows()
    tradeoff_rows = _compute_tradeoff_rows(summary_rows)
    save_results_csv(tradeoff_rows, TRADEOFF_CSV)

    logger.info("All experiments finished.")
    logger.info("Summary: %s", os.path.abspath(SUMMARY_CSV))
    logger.info("Tradeoff: %s", os.path.abspath(TRADEOFF_CSV))


if __name__ == "__main__":
    main()
