import argparse
import numpy as np
import pandas as pd

REQUIRED_METHODS = ["baseline", "single_selfhelp", "multi_selfhelp", "no_debias_agent"]
REQUIRED_BIASES = ["anchoring", "framing"]


def is_better(a: float, b: float, bias: str) -> bool:
    if bias == "anchoring":
        # Higher avg_confidence_d is better.
        return a > b
    # For framing, lower |delta| is better.
    return abs(a) < abs(b)


def compare_pair(df: pd.DataFrame, method_a: str, method_b: str) -> dict:
    result = {}
    for bias in REQUIRED_BIASES:
        sub = df[(df["bias"] == bias) & (df["method"].isin([method_a, method_b]))]
        if len(sub) != 2:
            result[bias] = {"status": "missing", "a": np.nan, "b": np.nan}
            continue

        a = float(sub[sub["method"] == method_a]["metric_value"].iloc[0])
        b = float(sub[sub["method"] == method_b]["metric_value"].iloc[0])

        if a == b:
            status = "tie"
        elif is_better(a, b, bias):
            status = "a_better"
        else:
            status = "a_worse"

        result[bias] = {"status": status, "a": a, "b": b}
    return result


def decide_question(pair_result: dict) -> str:
    statuses = [pair_result[b]["status"] for b in REQUIRED_BIASES]
    if "missing" in statuses:
        return "INCONCLUSIVE (missing rows)"
    if all(s == "a_better" for s in statuses):
        return "PASS"
    if any(s == "a_worse" for s in statuses):
        return "FAIL"
    if any(s == "a_better" for s in statuses):
        return "WEAK PASS (better on one, tie on the other)"
    return "INCONCLUSIVE (all ties)"


def run_diagnostics(df: pd.DataFrame):
    notes = []

    for bias in REQUIRED_BIASES:
        methods = sorted(df[df["bias"] == bias]["method"].unique().tolist())
        missing = sorted(set(REQUIRED_METHODS) - set(methods))
        if missing:
            notes.append(f"[coverage] {bias}: missing methods -> {missing}")

    for bias in REQUIRED_BIASES:
        vals = df[df["bias"] == bias]["metric_value"].astype(float).values
        if len(vals) > 0 and np.allclose(vals, vals[0]):
            notes.append(f"[signal] {bias}: all metric values identical ({vals[0]:.6f})")

    if np.allclose(df["metric_value"].astype(float).values, 0.0):
        notes.append("[signal] all metric values are 0.0 -> likely weak model signal or parse-limited outputs")

    cost = (
        df.groupby(["bias", "method"], as_index=False)[["tokens_used", "num_calls", "runtime_seconds"]]
        .mean()
        .sort_values(["bias", "runtime_seconds"], ascending=[True, False])
    )
    return notes, cost


def print_pair(title: str, pair_result: dict, a_name: str, b_name: str):
    print(f"\n{title}")
    for bias in REQUIRED_BIASES:
        r = pair_result[bias]
        if r["status"] == "missing":
            print(f"- {bias}: missing data")
            continue
        if bias == "anchoring":
            print(
                f"- {bias}: {a_name}={r['a']:.6f}, {b_name}={r['b']:.6f} "
                f"(higher is better) -> {r['status']}"
            )
        else:
            print(
                f"- {bias}: |{a_name}|={abs(r['a']):.6f}, |{b_name}|={abs(r['b']):.6f} "
                f"(lower is better) -> {r['status']}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Q1/Q2 outcomes from summary.csv"
    )
    parser.add_argument(
        "--summary",
        default="results/demo/summary.csv",
        help="Path to summary.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.summary)

    required_cols = {
        "model",
        "bias",
        "method",
        "metric_name",
        "metric_value",
        "tokens_used",
        "num_calls",
        "runtime_seconds",
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    q1 = compare_pair(df, "multi_selfhelp", "single_selfhelp")
    q2 = compare_pair(df, "multi_selfhelp", "no_debias_agent")

    q1_decision = decide_question(q1)
    q2_decision = decide_question(q2)

    print("=== DIRECT QUESTION OUTPUT ===")
    print(f"Q1 (multi-agent vs single-agent self-help): {q1_decision}")
    print(f"Q2 (architecture ablation, multi vs no_debias): {q2_decision}")

    print_pair("Q1 detail", q1, "multi_selfhelp", "single_selfhelp")
    print_pair("Q2 detail", q2, "multi_selfhelp", "no_debias_agent")

    notes, cost = run_diagnostics(df)

    print("\n=== DIAGNOSTICS ===")
    if notes:
        for note in notes:
            print(f"- {note}")
    else:
        print("- No major issues detected.")

    print("\n=== COST SNAPSHOT (mean by bias/method) ===")
    print(cost.to_string(index=False))


if __name__ == "__main__":
    main()
