"""
main.py
--------
Entry point for the Multi-Agent Cognitive Bias Framework.

Usage:
    python main.py                         # run everything
    python main.py --bias anchoring        # anchoring only
    python main.py --bias framing          # framing only
    python main.py --model facebook/opt-1.3b
    python main.py --mitigation selfhelp   # one strategy only
    python main.py --mitigation all        # all strategies
    python main.py --quick                 # fast test run (tiny model, few students)

Multi-Agent Pipeline:
    Agent 1 — DecisionAgent:   admissions officer (evaluates students)
    Agent 2 — DebiasingAgent:  rewrites biased prompts (self-help mitigation)
    Agent 3 — EvaluatorAgent:  audits decision patterns for bias
"""

import argparse
import logging
import os
import time

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Agent LLM Cognitive Bias Framework"
    )
    parser.add_argument(
        "--bias",
        choices=["anchoring", "framing", "both"],
        default="both",
        help="Which bias to test (default: both)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name (overrides config.py)",
    )
    parser.add_argument(
        "--mitigation",
        choices=[
            "baseline",
            "awareness",
            "contrastive",
            "counterfactual",
            "selfhelp",
            "single_selfhelp",
            "multi_selfhelp",
            "no_debias_agent",
            "all",
        ],
        default="all",
        help="Mitigation strategy to run (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: use tiny model, very few students",
    )
    parser.add_argument(
        "--no-evaluator",
        action="store_true",
        help="Skip the evaluator agent (faster)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (for gated models like Llama-2)",
    )
    parser.add_argument(
        "--n-students",
        type=int,
        default=None,
        help="Override student count for both anchoring and framing",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=None,
        help="Override anchoring order permutations",
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
        help="Override model temperature (default from config)",
    )
    parser.add_argument(
        "--log-cost",
        action="store_true",
        help="Write per-condition model-call and token usage logs",
    )
    return parser.parse_args()


def build_agents(model_name: str, hf_token=None, use_evaluator: bool = True):
    """Instantiate all three agents sharing the same model."""
    from decision_agent import DecisionAgent
    from debiasing_agent import DebiasingAgent
    from evaluator_agent import EvaluatorAgent
    import config

    token = hf_token or config.HF_TOKEN

    logger.info(f"Initialising agents with model: {model_name}")
    logger.info("(Models are lazy-loaded — first inference call will load weights)")

    decision_agent = DecisionAgent(
        model_name=model_name,
        device=config.DEVICE,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        hf_token=token,
    )

    debiasing_agent = DebiasingAgent(
        model_name=model_name,
        device=config.DEVICE,
        temperature=config.TEMPERATURE,
        hf_token=token,
    )

    evaluator_agent = None
    if use_evaluator:
        evaluator_agent = EvaluatorAgent(
            model_name=model_name,
            device=config.DEVICE,
            temperature=config.TEMPERATURE,
            hf_token=token,
        )

    # Share the same underlying pipeline to avoid loading weights twice
    # (agents will each call _load() but the pipeline is stored on the object)
    logger.info("Agents created. Note: all agents share the same base model.")

    return decision_agent, debiasing_agent, evaluator_agent


def run_anchoring(
    decision_agent,
    debiasing_agent,
    evaluator_agent,
    mitigations: list,
    n_students: int,
    n_permutations: int,
    seed: int,
    results_dir: str,
    model_name: str,
):
    """Run the full anchoring bias experiment."""
    from anchoring import AnchoringExperiment
    from student_profiles import generate_sequential_student_set
    from evaluate import compare_mitigations, save_results_csv, save_per_student_csv
    from visualize import (
        plot_anchoring_confidence, plot_per_student_confidence, print_results_table
    )

    logger.info(f"\n{'='*60}")
    logger.info("ANCHORING BIAS EXPERIMENT")
    logger.info(f"Students: {n_students} | Permutations: {n_permutations}")
    logger.info(f"Mitigations: {mitigations}")
    logger.info(f"{'='*60}")

    profiles = generate_sequential_student_set(n_students, seed=seed)
    experiment = AnchoringExperiment(decision_agent, debiasing_agent, evaluator_agent)

    results_by_mitigation = {}
    for mitigation in mitigations:
        logger.info(f"\n--- Running anchoring [{mitigation}] ---")
        result = experiment.run(
            profiles=profiles,
            n_permutations=n_permutations,
            seed=seed,
            mitigation=mitigation,
        )
        results_by_mitigation[mitigation] = result

        # Save per-student CSV
        save_per_student_csv(
            result,
            os.path.join(results_dir, f"anchoring_{mitigation}_per_student.csv")
        )

        # Print evaluator analysis if available
        if result.get("evaluator_analysis"):
            logger.info(f"Evaluator analysis:\n{result['evaluator_analysis']}")

    # Aggregate comparison
    summary_rows = compare_mitigations(results_by_mitigation, "anchoring")
    save_results_csv(
        summary_rows,
        os.path.join(results_dir, "anchoring_summary.csv")
    )
    print_results_table(summary_rows, "ANCHORING BIAS RESULTS")

    # Plot
    plots_dir = os.path.join(results_dir, "plots")
    plot_anchoring_confidence(
        summary_rows,
        os.path.join(plots_dir, "anchoring_confidence.png"),
        model_name=model_name,
    )

    # Per-student plot for baseline
    if "baseline" in results_by_mitigation:
        plot_per_student_confidence(
            results_by_mitigation["baseline"]["per_student"],
            mitigation="baseline",
            output_path=os.path.join(plots_dir, "anchoring_per_student_baseline.png"),
            model_name=model_name,
        )

    return results_by_mitigation


def run_framing(
    decision_agent,
    debiasing_agent,
    evaluator_agent,
    mitigations: list,
    n_students: int,
    seed: int,
    results_dir: str,
    model_name: str,
):
    """Run the full framing bias experiment."""
    from framing import FramingExperiment
    from student_profiles import generate_student_profiles
    from evaluate import compare_mitigations, save_results_csv, save_per_student_csv
    from visualize import plot_framing_delta, print_results_table

    logger.info(f"\n{'='*60}")
    logger.info("FRAMING BIAS EXPERIMENT")
    logger.info(f"Students: {n_students}")
    logger.info(f"Mitigations: {mitigations}")
    logger.info(f"{'='*60}")

    profiles = generate_student_profiles(n_students, seed=seed)
    experiment = FramingExperiment(decision_agent, debiasing_agent, evaluator_agent)

    results_by_mitigation = {}
    for mitigation in mitigations:
        logger.info(f"\n--- Running framing [{mitigation}] ---")
        result = experiment.run(profiles=profiles, mitigation=mitigation)
        results_by_mitigation[mitigation] = result

        # Save per-student CSV
        save_per_student_csv(
            result,
            os.path.join(results_dir, f"framing_{mitigation}_per_student.csv")
        )

        # Print evaluator analysis if available
        if result.get("evaluator_analysis"):
            logger.info(f"Evaluator analysis:\n{result['evaluator_analysis']}")

    # Aggregate comparison
    summary_rows = compare_mitigations(results_by_mitigation, "framing")
    save_results_csv(
        summary_rows,
        os.path.join(results_dir, "framing_summary.csv")
    )
    print_results_table(summary_rows, "FRAMING BIAS RESULTS")

    # Plot
    plots_dir = os.path.join(results_dir, "plots")
    plot_framing_delta(
        summary_rows,
        os.path.join(plots_dir, "framing_delta.png"),
        model_name=model_name,
    )

    return results_by_mitigation


def main():
    args = parse_args()

    import config

    # ── Override config from CLI ───────────────────────────────────────────────
    model_name = args.model or config.MODEL_NAME
    if args.quick:
        model_name = "facebook/opt-125m"
        config.NUM_STUDENTS_ANCHORING = 4
        config.NUM_ORDER_PERMUTATIONS = 3
        config.NUM_STUDENTS_FRAMING   = 6
        logger.info("QUICK MODE: using opt-125m, minimal students")

    if args.hf_token:
        config.HF_TOKEN = args.hf_token
    if args.temperature is not None:
        config.TEMPERATURE = args.temperature
    if args.n_students is not None:
        config.NUM_STUDENTS_ANCHORING = args.n_students
        config.NUM_STUDENTS_FRAMING = args.n_students
    if args.n_permutations is not None:
        config.NUM_ORDER_PERMUTATIONS = args.n_permutations
    if args.seed is not None:
        config.RANDOM_SEED = args.seed

    mitigations = (
        config.MITIGATIONS_TO_RUN
        if args.mitigation == "all"
        else [args.mitigation]
    )

    run_biases = []
    if args.bias in ("anchoring", "both"):
        run_biases.append("anchoring")
    if args.bias in ("framing", "both"):
        run_biases.append("framing")

    use_evaluator = not args.no_evaluator

    # ── Create output dirs ─────────────────────────────────────────────────────
    results_dir = config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)

    # ── Build agents ───────────────────────────────────────────────────────────
    decision_agent, debiasing_agent, evaluator_agent = build_agents(
        model_name=model_name,
        hf_token=config.HF_TOKEN,
        use_evaluator=use_evaluator,
    )

    # ── Run experiments ────────────────────────────────────────────────────────
    all_results = {}
    cost_rows = []

    def _reset_agent_stats():
        for agent in (decision_agent, debiasing_agent, evaluator_agent):
            if agent and hasattr(agent, "reset_usage_stats"):
                agent.reset_usage_stats()

    def _collect_cost_rows(bias_name: str, mitigation: str, wall_runtime: float):
        agent_map = {
            "decision": decision_agent,
            "debiasing": debiasing_agent,
            "evaluator": evaluator_agent,
        }
        total_tokens = 0
        total_calls = 0
        for role, agent in agent_map.items():
            if not agent or not hasattr(agent, "get_usage_stats"):
                continue
            stats = agent.get_usage_stats()
            total_role_tokens = stats["input_tokens"] + stats["output_tokens"]
            total_tokens += total_role_tokens
            total_calls += stats["num_calls"]
            cost_rows.append({
                "bias": bias_name,
                "mitigation": mitigation,
                "agent_role": role,
                "num_calls": stats["num_calls"],
                "input_tokens": stats["input_tokens"],
                "output_tokens": stats["output_tokens"],
                "tokens_used": total_role_tokens,
                "runtime_seconds": round(stats["runtime_seconds"], 4),
                "wall_runtime_seconds": round(wall_runtime, 4),
            })
        cost_rows.append({
            "bias": bias_name,
            "mitigation": mitigation,
            "agent_role": "total",
            "num_calls": total_calls,
            "input_tokens": "",
            "output_tokens": "",
            "tokens_used": total_tokens,
            "runtime_seconds": "",
            "wall_runtime_seconds": round(wall_runtime, 4),
        })

    if "anchoring" in run_biases:
        if args.log_cost and len(mitigations) == 1:
            _reset_agent_stats()
            start = time.perf_counter()
        all_results["anchoring"] = run_anchoring(
            decision_agent=decision_agent,
            debiasing_agent=debiasing_agent,
            evaluator_agent=evaluator_agent,
            mitigations=mitigations,
            n_students=config.NUM_STUDENTS_ANCHORING,
            n_permutations=config.NUM_ORDER_PERMUTATIONS,
            seed=config.RANDOM_SEED,
            results_dir=results_dir,
            model_name=model_name,
        )
        if args.log_cost and len(mitigations) == 1:
            _collect_cost_rows(
                bias_name="anchoring",
                mitigation=mitigations[0],
                wall_runtime=time.perf_counter() - start,
            )

    if "framing" in run_biases:
        if args.log_cost and len(mitigations) == 1:
            _reset_agent_stats()
            start = time.perf_counter()
        all_results["framing"] = run_framing(
            decision_agent=decision_agent,
            debiasing_agent=debiasing_agent,
            evaluator_agent=evaluator_agent,
            mitigations=mitigations,
            n_students=config.NUM_STUDENTS_FRAMING,
            seed=config.RANDOM_SEED,
            results_dir=results_dir,
            model_name=model_name,
        )
        if args.log_cost and len(mitigations) == 1:
            _collect_cost_rows(
                bias_name="framing",
                mitigation=mitigations[0],
                wall_runtime=time.perf_counter() - start,
            )

    if args.log_cost and cost_rows:
        from evaluate import save_results_csv
        save_results_csv(cost_rows, os.path.join(results_dir, "cost_summary.csv"))
        logger.info("Saved cost summary to results/cost_summary.csv")

    logger.info(f"\nAll results saved to: {os.path.abspath(results_dir)}/")
    logger.info("Done.")
    return all_results


if __name__ == "__main__":
    main()
