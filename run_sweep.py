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
import io, contextlib

sys.path.insert(0, ".")
_mod = importlib.import_module("bridgingNorms")
_original_run_task = _mod.CommunityNormAnalyzer.run_task

def run_task(self, df, community_a, community_b, task_name,
                      n_samples, joint=True, verbose=True, _reason="sweep"):
    import builtins
    _real_input = builtins.input
    builtins.input = lambda _prompt="": _reason
    try:
        result = _original_run_task(
            self, df, community_a, community_b, task_name,
            n_samples, joint=joint, verbose=verbose
        )
    finally:
        builtins.input = _real_input
    return result

_mod.CommunityNormAnalyzer.run_task = run_task

CommunityNormAnalyzer = _mod.CommunityNormAnalyzer
NotEnoughSamplesError = _mod.NotEnoughSamplesError

# parameters: eventually should be on input or automatic, for now just set here for ease
DEFAULT_SIZES   = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
DEFAULT_TRIALS  = 10
DATA_FILE       = "data_training_selected_clusters_comments_and_rules.csv"
BASE_METRICS = ["coverage", "violating_fraction", "unique_comments", "unique_comments_a", "unique_comments_b"]

def parse_args():
    parser = argparse.ArgumentParser(description="Sweep sample sizes for bridgingNorms")
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
def run_single(analyzer, df, community_a, community_b, n_samples, reason, verbose):
    """Run one trial. Returns a flat metrics dict, or None on failure."""
    captured = io.StringIO()
    ctx = contextlib.redirect_stdout(captured) if not verbose else contextlib.nullcontext()

    try:
        with ctx:
            result = analyzer.run_task(
                df,
                community_a=community_a,
                community_b=community_b,
                task_name="task2",
                n_samples=n_samples,
                verbose=verbose,
                _reason=reason,
            )
        
        if not result or "error" in result:
            print("Run failed:", result.get("error", "None"))
            return None

        if "metrics" not in result or result["metrics"] is None:
            return None
        m = result["metrics"].copy()
        m["community_a"] = community_a
        m["community_b"] = community_b
        m["n_samples"]   = n_samples
        m["prompt"]          = result["prompt"]
        m["raw_response"]    = result["raw_response"]
        m["input_comments"]  = [
            {"comment_id": cid, "text": text, "community": comm, "status": status}
            for cid, text, comm, status in result["input_comments"]
        ]
        m["run_log"]     = captured.getvalue()
        return m

    except NotEnoughSamplesError as e:
        print(f"    Skipped — not enough samples: {e}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def save_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_csv(records, path):
    if not records:
        return
    fieldnames = sorted({k for r in records for k in r})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def plot_metrics(records, out_dir: Path):
    if not records:
        print("No records to plot.")
        return

    sizes = sorted({r["n_samples"] for r in records})
    palette = plt.cm.tab10.colors

    metrics_to_plot = [m for m in BASE_METRICS
                       if m not in ('unique_comments', 'unique_comments_a', 'unique_comments_b')
                       and any(m in r and r[m] is not None for r in records)]

    n = len(metrics_to_plot) + 1
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes_flat[idx]
        xs_mean, ys_mean, ys_std = [], [], []

        for size in sizes:
            vals = [r[metric] for r in records
                    if r["n_samples"] == size and r.get(metric) is not None]
            if not vals:
                continue
            xs_mean.append(size)
            ys_mean.append(float(np.mean(vals)))
            ys_std.append(float(np.std(vals)))

        if xs_mean:
            ys_mean_arr = np.array(ys_mean)
            ys_std_arr  = np.array(ys_std)
            ax.fill_between(xs_mean, ys_mean_arr - ys_std_arr, ys_mean_arr + ys_std_arr,
                            color=palette[0], alpha=0.18, zorder=2)
            ax.plot(xs_mean, ys_mean_arr, "o-", linewidth=2.2, markersize=8,
                    color=palette[0], zorder=4)

        pretty = metric.replace("_", " ").title()
        ax.set_title(pretty, fontsize=13, fontweight="bold", pad=8)
        ax.set_xlabel("n_samples", fontsize=11)
        ax.set_ylabel(pretty, fontsize=11)
        ax.xaxis.set_major_locator(ticker.FixedLocator(sizes))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(bottom=0)
        ax.set_ylim(top=1.0)

    ax = axes_flat[len(metrics_to_plot)]
    for label, metric, color, marker in [
        ('Both communities', 'unique_comments',   palette[3], 'o'),
        ('Community A',      'unique_comments_a', palette[0], 's'),
        ('Community B',      'unique_comments_b', palette[2], '^'),
    ]:
        xs_mean, ys_mean, ys_std = [], [], []
        for size in sizes:
            vals = [r[metric] for r in records
                    if r["n_samples"] == size and r.get(metric) is not None]
            if not vals:
                continue
            xs_mean.append(size)
            ys_mean.append(float(np.mean(vals)))
            ys_std.append(float(np.std(vals)))
        if xs_mean:
            ys_mean_arr = np.array(ys_mean)
            ys_std_arr  = np.array(ys_std)
            ax.fill_between(xs_mean, ys_mean_arr - ys_std_arr, ys_mean_arr + ys_std_arr,
                            color=color, alpha=0.18, zorder=2)
            ax.plot(xs_mean, ys_mean_arr, f"{marker}-", linewidth=2.2, markersize=8,
                color=color, zorder=4, label=label)

    ax.set_title("Unique Comments", fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("n_samples", fontsize=11)
    ax.set_ylabel("Unique Comments", fontsize=11)
    ax.xaxis.set_major_locator(ticker.FixedLocator(sizes))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=200)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Metrics vs Sample Size ",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    plot_path = out_dir / f"sweep_metrics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {plot_path}")
    plt.close(fig)


def main():
    args      = parse_args()
    sizes     = sorted(args.sizes)
    trials    = args.trials
    reason    = args.reason
    verbose   = args.verbose

    print(f"\n{'='*60}")
    print(f"  Sizes  :  {sizes}")
    print(f"  Trials :  {trials} per size")
    print(f"  Reason :  {reason}")
    print(f"{'='*60}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(f"sweep_{timestamp}")
    out_dir.mkdir(exist_ok=True)

    analyzer = CommunityNormAnalyzer()
    print(f"Loading data …")
    df = analyzer.load_data(args.data, min_samp=1)
    print()

    all_records = []

    for n_samples in sizes:
        min_samp = n_samples // 2
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

        while trial_count < trials and attempts < max_attempts:
            attempts += 1
            community_a, community_b = random.sample(eligible, 2)
            print(f"  Trial {trial_count + 1}/{trials}: {community_a}  vs  {community_b}")

            metrics = run_single(
                analyzer, df, community_a, community_b, n_samples, reason, verbose
            )
            if metrics is not None:
                all_records.append(metrics)
                trial_count += 1
                print(f"    trial {trial_count}/{trials}  "
                        f"cov={metrics.get('coverage', float('nan')):.3f}  "
                        f"vf={metrics.get('violating_fraction', float('nan')):.3f}")

        if trial_count < trials:
            print(f"  Only {trial_count}/{trials} trials completed for n_samples={n_samples}")

    jsonl_path = out_dir / f"sweep_results.jsonl"
    csv_path   = out_dir / f"sweep_results.csv"
    save_jsonl(all_records, jsonl_path)
    save_csv(all_records,   csv_path)
    print(f"\nResults  → {jsonl_path}")
    print(f"Results  → {csv_path}")

    plot_metrics(all_records, out_dir)

    print(f"\nAll outputs saved to: {out_dir}/\n")

if __name__ == "__main__":
    main()
