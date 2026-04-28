
   # python run_sweep.py <task_name> <num_norms> [--sizes 5 10 20] [--trials 3] [--reason "my reason"]

import sys
import json
import random
import argparse
import csv
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import importlib, sys

sys.path.insert(0, ".")
_mod = importlib.import_module("bridgingNorms")
_original_run_task = _mod.CommunityNormAnalyzer.run_task

def run_task(self, df, community_a, community_b, task_name,
                      n_samples, num_norms, joint=True, verbose=True, _reason="sweep"):
    import builtins
    _real_input = builtins.input
    builtins.input = lambda _prompt="": _reason
    try:
        result = _original_run_task(
            self, df, community_a, community_b, task_name,
            n_samples, num_norms, joint=joint, verbose=verbose
        )
    finally:
        builtins.input = _real_input
    return result

_mod.CommunityNormAnalyzer.run_task = run_task

CommunityNormAnalyzer = _mod.CommunityNormAnalyzer
NotEnoughSamplesError = _mod.NotEnoughSamplesError

# parameters: eventually should be on input or automatic, for now just set here for ease
DEFAULT_SIZES   = [10, 20, 30, 40, 50, 60, 70]
DEFAULT_TRIALS  = 2
DATA_FILE       = "data_training_selected_clusters_comments_and_rules.csv"
BASE_METRICS    = ["coverage", "redundancy", "violating_fraction", "unique_comments"]

# pretty straightforward, parse args to use for metrics
def parse_args():
    parser = argparse.ArgumentParser(description="Sweep sample sizes for bridgingNorms")
    parser.add_argument("task_name", help="task1 or task2")
    parser.add_argument("num_norms", type=int, help="Number of norms to extract")
    parser.add_argument("--sizes",  type=int, nargs="+", default=DEFAULT_SIZES,
                        help=f"Sample sizes to sweep (default: {DEFAULT_SIZES})")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Trials per sample size with different community pairs (default: {DEFAULT_TRIALS})")
    parser.add_argument("--reason", type=str, default="automated sweep",
                        help="Reason logged in each run")
    parser.add_argument("--data",   type=str, default=DATA_FILE,
                        help=f"Path to CSV dataset (default: {DATA_FILE})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print prompts and LLM responses")
    return parser.parse_args()


# one single trial with all the inputs, print saves some to find similarity
def run_single(analyzer, df, community_a, community_b,
               task_name, n_samples, num_norms, reason, verbose):
    """Run one trial. Returns a flat metrics dict, or None on failure."""
    try:
        result = analyzer.run_task(
            df,
            community_a=community_a,
            community_b=community_b,
            task_name=task_name,
            n_samples=n_samples,
            num_norms=num_norms,
            verbose=verbose,
            _reason=reason,
        )
        if not result:
            return None

        m = result["metrics"].copy()
        m["community_a"] = community_a
        m["community_b"] = community_b
        m["n_samples"]   = n_samples

        # if task_name == "task2":
        #     m["definition_similarity"] = (
        #         result["metrics"].get("inputs", {}).get("definition_similarity")
        #     )
        return m

    except NotEnoughSamplesError as e:
        print(f"    Skipped — not enough samples: {e}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


# save json
def save_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

#save csv
def save_csv(records, path):
    if not records:
        return
    fieldnames = sorted({k for r in records for k in r})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

# plot everything
def plot_metrics(records, task_name, out_dir: Path):
    """One subplot per metric: mean ± std line + individual trial scatter."""
    if not records:
        print("No records to plot.")
        return

    sizes = sorted({r["n_samples"] for r in records})

    metrics_to_plot = BASE_METRICS.copy()
    # if task_name == "task2":
    #     metrics_to_plot.append("definition_similarity")
    
    metrics_to_plot = [m for m in metrics_to_plot
                       if any(m in r and r[m] is not None for r in records)]

    n = len(metrics_to_plot)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    palette   = plt.cm.tab10.colors

    rng = np.random.default_rng(42)

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes_flat[idx]
        color = palette[idx % len(palette)]
        xs_mean, ys_mean, ys_std = [], [], []

        for size in sizes:
            vals = [r[metric] for r in records
                    if r["n_samples"] == size and r.get(metric) is not None]
            if not vals:
                continue
            xs_mean.append(size)
            ys_mean.append(float(np.mean(vals)))
            ys_std.append(float(np.std(vals)))

            jitter = rng.uniform(-size * 0.04, size * 0.04, len(vals))
            ax.scatter([size + j for j in jitter], vals,
                       alpha=0.5, s=45, color=color, zorder=3, label="_nolegend_")

        if xs_mean:
            ax.errorbar(xs_mean, ys_mean, yerr=ys_std,
                        fmt="o-", linewidth=2.2, markersize=8,
                        color=color, capsize=5, zorder=4, label="mean ± std")

        pretty = metric.replace("_", " ").title()
        ax.set_title(pretty, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("n_samples", fontsize=11)
        ax.set_ylabel(pretty, fontsize=11)
        ax.xaxis.set_major_locator(ticker.FixedLocator(sizes))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=9)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Metrics vs Sample Size  —  {task_name}",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    plot_path = out_dir / f"{task_name}_sweep_metrics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {plot_path}")
    plt.close(fig)

# Add this function after plot_metrics()

def plot_community_metrics(records, task_name, out_dir: Path):
    """One figure per community pair: metrics across sample sizes."""
    if not records:
        return

    # Group records by community pair (order-independent)
    from collections import defaultdict
    pairs = defaultdict(list)
    for r in records:
        key = tuple(sorted([r["community_a"], r["community_b"]]))
        pairs[key].append(r)

    metrics_to_plot = BASE_METRICS.copy()
    # if task_name == "task2":
    #     metrics_to_plot.append("definition_similarity")
    metrics_to_plot = [m for m in metrics_to_plot
                       if any(m in r and r[m] is not None for r in records)]

    n_metrics = len(metrics_to_plot)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    palette = plt.cm.tab10.colors

    for (comm_a, comm_b), pair_records in pairs.items():
        sizes = sorted({r["n_samples"] for r in pair_records})

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
        axes_flat = np.array(axes).flatten() if n_metrics > 1 else [axes]

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes_flat[idx]
            color = palette[idx % len(palette)]
            xs, ys = [], []

            for size in sizes:
                vals = [r[metric] for r in pair_records
                        if r["n_samples"] == size and r.get(metric) is not None]
                for v in vals:
                    xs.append(size)
                    ys.append(float(v))

            if xs:
                ax.scatter(xs, ys, color=color, s=60, zorder=3, alpha=0.8)
                ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.5, zorder=2)

            pretty = metric.replace("_", " ").title()
            ax.set_title(pretty, fontsize=13, fontweight="bold", pad=8)
            ax.set_xlabel("n_samples", fontsize=11)
            ax.set_ylabel(pretty, fontsize=11)
            ax.xaxis.set_major_locator(ticker.FixedLocator(sizes if sizes else [0]))
            ax.grid(axis="y", linestyle="--", alpha=0.35)

        for ax in axes_flat[n_metrics:]:
            ax.set_visible(False)

        safe_name = f"{comm_a}_vs_{comm_b}".replace("/", "-").replace(" ", "_")
        fig.suptitle(f"{comm_a}  vs  {comm_b}  —  {task_name}",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()

        plot_path = out_dir / f"{safe_name}_metrics.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Community plot → {plot_path}")
        plt.close(fig)

def main():
    args      = parse_args()
    task_name = args.task_name
    num_norms = args.num_norms
    sizes     = sorted(args.sizes)
    trials    = args.trials
    reason    = args.reason
    verbose   = args.verbose

# print everything out, maybe move to json or remove later?
    print(f"\n{'='*60}")
    print(f"  Sweep  :  {task_name}  |  norms = {num_norms}")
    print(f"  Sizes  :  {sizes}")
    print(f"  Trials :  {trials} per size")
    print(f"  Reason :  {reason}")
    print(f"{'='*60}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(f"sweep_{task_name}_{timestamp}")
    out_dir.mkdir(exist_ok=True)

    analyzer = CommunityNormAnalyzer()
    print(f"Loading data …")
    df = analyzer.load_data(args.data, min_samp=1)
    print()

    all_records = []

    for n_samples in sizes:
        min_samp = n_samples // 2
        # filtering out groups without enough violations/non-violations
        counts = df.groupby("community_id")["violation"].agg(
            violations=lambda x: x.sum(),
            non_violations=lambda x: (~x).sum()
        )
        eligible = counts[
            (counts["violations"] >= min_samp) &
            (counts["non_violations"] >= min_samp)
        ].index.tolist()

        print(f"\n── n_samples = {n_samples}  (eligible communities: {len(eligible)}) {'─'*20}")

        if len(eligible) < 2:
            print(f"  Skipping — fewer than 2 communities have {min_samp} samples per class")
            continue

        trial_count = 0
        attempts    = 0
        max_attempts = trials * 15
        # randomly tries to run with 2 communities, if it doesn't work, try again until we reach threshold
        while trial_count < trials and attempts < max_attempts:
            attempts += 1
            community_a, community_b = random.sample(eligible, 2)
            print(f"  Trial {trial_count + 1}/{trials}: {community_a}  vs  {community_b}")

            metrics = run_single(
                analyzer, df, community_a, community_b,
                task_name, n_samples, num_norms, reason, verbose
            )
            #if successful, append data and keep going
            if metrics is not None:
                all_records.append(metrics)
                trial_count += 1
                cov  = metrics.get("coverage", float("nan"))
                red  = metrics.get("redundancy", float("nan"))
                vf   = metrics.get("violating_fraction", float("nan"))
                print(f"    coverage={cov:.3f}  redundancy={red:.3f}  violating_fraction={vf:.3f}")

        if trial_count < trials:
            print(f"  Only {trial_count}/{trials} trials completed for n_samples={n_samples}")

    # raw results
    jsonl_path = out_dir / f"{task_name}_sweep_results.jsonl"
    csv_path   = out_dir / f"{task_name}_sweep_results.csv"
    save_jsonl(all_records, jsonl_path)
    save_csv(all_records,   csv_path)
    print(f"\nResults  → {jsonl_path}")
    print(f"Results  → {csv_path}")

    # plot
    plot_metrics(all_records, task_name, out_dir)
    plot_community_metrics(all_records, task_name, out_dir)

    print(f"\nAll outputs saved to: {out_dir}/\n")


if __name__ == "__main__":
    main()