"""
evaluate.py
-----------
Compute and aggregate bias metrics matching the BiasBuster paper.

Anchoring:  normalized Euclidean distance d (Eq. 1)
Framing:    Δ admit rate = admit_rate(admit_frame) - admit_rate(reject_frame)
"""

import math
import csv
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ── Anchoring metrics ─────────────────────────────────────────────────────────

def compute_anchoring_confidence(
    r_instance: float, r_overall: float
) -> float:
    """
    Equation 1 from the paper.
    d(Si, A) = sqrt( (r_instance - r_overall)^2 + ((1-r_instance) - (1-r_overall))^2 )
    normalized by sqrt(2).
    """
    d = math.sqrt(
        (r_instance - r_overall) ** 2 +
        ((1 - r_instance) - (1 - r_overall)) ** 2
    )
    return d / math.sqrt(2)


def summarize_anchoring_results(result: Dict) -> Dict:
    """
    Return a clean summary of anchoring experiment results.
    """
    return {
        "mitigation": result["mitigation"],
        "n_students": result["n_students"],
        "n_permutations": result["n_permutations"],
        "overall_admission_rate": round(result["overall_admission_rate"], 4),
        "avg_confidence_d": round(result["avg_confidence_d"], 4),
    }


# ── Framing metrics ───────────────────────────────────────────────────────────

def summarize_framing_results(result: Dict) -> Dict:
    """
    Return a clean summary of framing experiment results.
    """
    return {
        "mitigation": result["mitigation"],
        "n_students": result["n_students"],
        "admit_rate_admit_frame": round(result["admit_rate_admit_frame"], 4),
        "admit_rate_reject_frame": round(result["admit_rate_reject_frame"], 4),
        "admit_rate_neutral": round(result["admit_rate_neutral"], 4),
        "delta": round(result["delta"], 4),
        "n_inconsistent": result["n_inconsistent"],
        "pct_inconsistent": round(
            result["n_inconsistent"] / max(result["n_students"], 1), 4
        ),
    }


# ── Aggregate across mitigations ──────────────────────────────────────────────

def compare_mitigations(results_by_mitigation: Dict[str, Dict], bias: str) -> List[Dict]:
    """
    Given a dict of {mitigation_name: result_dict}, return a comparison table.
    bias: "anchoring" | "framing"
    """
    rows = []
    for mitigation, result in results_by_mitigation.items():
        if bias == "anchoring":
            rows.append(summarize_anchoring_results(result))
        elif bias == "framing":
            rows.append(summarize_framing_results(result))
    return rows


# ── CSV export ────────────────────────────────────────────────────────────────

def save_results_csv(rows: List[Dict], filepath: str):
    """Save a list of dicts to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not rows:
        logger.warning(f"No rows to save to {filepath}")
        return
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} rows → {filepath}")


def save_per_student_csv(result: Dict, filepath: str):
    """Save per-student details from an experiment result to CSV."""
    per_student = result.get("per_student", [])
    if not per_student:
        return

    # Flatten nested keys
    flat_rows = []
    for row in per_student:
        flat = {k: v for k, v in row.items() if not isinstance(v, dict)}
        flat["mitigation"] = result["mitigation"]
        flat_rows.append(flat)

    save_results_csv(flat_rows, filepath)
