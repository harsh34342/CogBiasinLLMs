"""
visualize.py
------------
Generate plots for bias experiment results.
Styled to match Figure 3 and Table 3 from the BiasBuster paper.
"""

import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for headless servers
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not installed — skipping plots")


def plot_framing_delta(
    mitigation_summaries: List[Dict],
    output_path: str,
    model_name: str = "",
):
    """
    Bar chart: Δ admit rate per mitigation strategy (framing bias).
    Δ = admit_rate(admit_frame) - admit_rate(reject_frame)
    Smaller absolute Δ = less bias.
    """
    if not HAS_MPL:
        return

    strategies = [r["mitigation"] for r in mitigation_summaries]
    deltas     = [abs(r["delta"]) for r in mitigation_summaries]
    admit_rates = [r["admit_rate_admit_frame"] for r in mitigation_summaries]
    reject_rates = [r["admit_rate_reject_frame"] for r in mitigation_summaries]

    x = range(len(strategies))
    width = 0.3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Δ per strategy
    ax = axes[0]
    bars = ax.bar(x, deltas, color=["#d62728" if s == "baseline" else "#1f77b4"
                                     for s in strategies])
    ax.set_xticks(list(x))
    ax.set_xticklabels(strategies, rotation=20, ha="right")
    ax.set_ylabel("|Δ Admit Rate|")
    ax.set_title(f"Framing Bias — |Δ| per Mitigation\n{model_name}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Right: admit rates per frame per strategy
    ax2 = axes[1]
    x2 = range(len(strategies))
    ax2.bar([i - width/2 for i in x2], admit_rates, width=width,
            label="Admit frame", color="#2ca02c", alpha=0.8)
    ax2.bar([i + width/2 for i in x2], reject_rates, width=width,
            label="Reject frame", color="#ff7f0e", alpha=0.8)
    ax2.set_xticks(list(x2))
    ax2.set_xticklabels(strategies, rotation=20, ha="right")
    ax2.set_ylabel("Admission Rate")
    ax2.set_ylim(0, 1.0)
    ax2.axhline(0.3, color="gray", linestyle="--", linewidth=0.8, label="Expected 30%")
    ax2.set_title(f"Framing Bias — Admit Rates by Frame\n{model_name}")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved framing plot → {output_path}")


def plot_anchoring_confidence(
    mitigation_summaries: List[Dict],
    output_path: str,
    model_name: str = "",
):
    """
    Bar chart: average decision confidence d per mitigation strategy.
    Higher d = more consistent decisions (less anchoring bias effect).
    """
    if not HAS_MPL:
        return

    strategies = [r["mitigation"] for r in mitigation_summaries]
    conf_scores = [r["avg_confidence_d"] for r in mitigation_summaries]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        range(len(strategies)),
        conf_scores,
        color=["#d62728" if s == "baseline" else "#1f77b4" for s in strategies],
    )
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=20, ha="right")
    ax.set_ylabel("Avg Confidence d (normalized Euclidean)")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Anchoring Bias — Decision Confidence per Mitigation\n{model_name}")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="d=0.5 reference")
    ax.legend(fontsize=8)

    for bar, val in zip(bars, conf_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved anchoring plot → {output_path}")


def plot_per_student_confidence(
    per_student: List[Dict],
    mitigation: str,
    output_path: str,
    model_name: str = "",
):
    """
    Per-student confidence d scores as a horizontal bar chart.
    """
    if not HAS_MPL:
        return

    ids = [f"S{r['student_id']}" for r in per_student]
    confs = [r["confidence_d"] for r in per_student]

    fig, ax = plt.subplots(figsize=(7, max(4, len(ids) * 0.4)))
    colors = ["#2ca02c" if c > 0.5 else "#d62728" for c in confs]
    ax.barh(ids, confs, color=colors)
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Decision Confidence d")
    ax.set_title(f"Per-Student Confidence — Anchoring [{mitigation}]\n{model_name}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved per-student plot → {output_path}")


def print_results_table(rows: List[Dict], title: str):
    """Print a nicely formatted table to stdout."""
    if not rows:
        return
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows))
                  for h in headers}
    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers))
    print()
