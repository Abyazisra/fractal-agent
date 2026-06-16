"""
Generate publication-quality figures from evaluation results.
Produces IEEE/ACM-style figures suitable for top-tier venues.

Usage: python generate_figures.py
"""

import json, os, sys
import numpy as np

RESULTS_DIR = "research_results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_results():
    path = os.path.join(RESULTS_DIR, "full_results.json")
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run research_eval.py first.")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)

def setup_matplotlib():
    """Configure matplotlib for publication-quality output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def fig1_rouge_comparison(results, plt):
    """Figure 1: Grouped bar chart — ROUGE scores across modes per document."""
    docs = list(results.keys())
    modes = ["base", "predictive", "differentiable"]
    mode_labels = ["Base ReadAgent", "Predictive Gisting", "Differentiable Retrieval"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

    fig, axes = plt.subplots(1, len(docs), figsize=(5 * len(docs), 3.5), sharey=True)
    if len(docs) == 1:
        axes = [axes]

    for ax_idx, doc in enumerate(docs):
        ax = axes[ax_idx]
        x = np.arange(len(metrics))
        width = 0.25
        for i, mode in enumerate(modes):
            if mode not in results[doc]["modes"]:
                continue
            avgs = results[doc]["modes"][mode]["averages"]
            vals = [avgs[m] for m in metrics]
            bars = ax.bar(x + i * width, vals, width, label=mode_labels[i],
                         color=colors[i], edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xlabel("")
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics)
        wc = results[doc].get("word_count", "?")
        ax.set_title(f"{doc.replace('_', ' ').title()}\n({wc} words)")
        if ax_idx == 0:
            ax.set_ylabel("F-Measure Score")

    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fancybox=False)
    fig.suptitle("ROUGE Score Comparison Across Pipeline Modes", fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig1_rouge_comparison.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


def fig2_llm_ratings(results, plt):
    """Figure 2: LLM Rating comparison (LR-1 strict, LR-2 permissive)."""
    modes = ["base", "predictive", "differentiable"]
    mode_labels = ["Base\nReadAgent", "Predictive\nGisting", "Differentiable\nRetrieval"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    # Aggregate across all documents
    lr1_vals, lr2_vals = [], []
    for mode in modes:
        lr1_all, lr2_all = [], []
        for doc in results:
            if mode in results[doc]["modes"]:
                lr1_all.append(results[doc]["modes"][mode]["averages"]["LR-1"])
                lr2_all.append(results[doc]["modes"][mode]["averages"]["LR-2"])
        lr1_vals.append(np.mean(lr1_all) if lr1_all else 0)
        lr2_vals.append(np.mean(lr2_all) if lr2_all else 0)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(modes))
    width = 0.35
    bars1 = ax.bar(x - width/2, lr1_vals, width, label="LR-1 (Strict)", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width/2, lr2_vals, width, label="LR-2 (Permissive)", color="#DD8452", edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Match Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels)
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, fancybox=False)
    ax.set_title("LLM Rating Comparison (Aggregated)", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig2_llm_ratings.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


def fig3_cr_vs_quality(results, plt):
    """Figure 3: Scatter — Compression Rate vs ROUGE-L (quality-efficiency tradeoff)."""
    colors = {"base": "#4C72B0", "predictive": "#DD8452", "differentiable": "#55A868"}
    markers = {"base": "o", "predictive": "s", "differentiable": "D"}
    labels = {"base": "Base ReadAgent", "predictive": "Predictive Gisting", "differentiable": "Differentiable Retrieval"}

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for doc in results:
        for mode in results[doc]["modes"]:
            avgs = results[doc]["modes"][mode]["averages"]
            ax.scatter(avgs["CR"], avgs["ROUGE-L"], c=colors.get(mode, "gray"),
                      marker=markers.get(mode, "o"), s=100, edgecolors="black", linewidth=0.5,
                      label=labels.get(mode, mode), zorder=5)
            ax.annotate(doc.split("_")[0], (avgs["CR"], avgs["ROUGE-L"]),
                       textcoords="offset points", xytext=(5, 5), fontsize=7)

    # Deduplicate legend
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True, fancybox=False)
    ax.set_xlabel("Compression Rate (%)")
    ax.set_ylabel("ROUGE-L F-Measure")
    ax.set_title("Quality–Efficiency Tradeoff", fontweight="bold")
    ax.invert_xaxis()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig3_cr_vs_quality.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


def fig4_token_efficiency(results, plt):
    """Figure 4: Token usage vs ROUGE-L — computational cost analysis."""
    colors = {"base": "#4C72B0", "predictive": "#DD8452", "differentiable": "#55A868"}
    labels = {"base": "Base ReadAgent", "predictive": "Predictive Gisting", "differentiable": "Differentiable Retrieval"}

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for doc in results:
        for mode in results[doc]["modes"]:
            avgs = results[doc]["modes"][mode]["averages"]
            ax.scatter(avgs["Tokens"], avgs["ROUGE-L"], c=colors.get(mode, "gray"),
                      s=120, edgecolors="black", linewidth=0.5,
                      label=labels.get(mode, mode), zorder=5, alpha=0.8)

    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), frameon=True, fancybox=False)
    ax.set_xlabel("Average Tokens Consumed")
    ax.set_ylabel("ROUGE-L F-Measure")
    ax.set_title("Token Efficiency Analysis", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig4_token_efficiency.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


def fig5_per_question_heatmap(results, plt):
    """Figure 5: Heatmap — per-question ROUGE-L across modes."""
    import matplotlib.colors as mcolors

    for doc in results:
        modes = list(results[doc]["modes"].keys())
        questions = []
        data_matrix = []
        for mode in modes:
            rougeL_vals = results[doc]["modes"][mode]["rougeL"]
            data_matrix.append(rougeL_vals)
            if not questions:
                questions = [f"Q{i+1}" for i in range(len(rougeL_vals))]

        data = np.array(data_matrix)
        fig, ax = plt.subplots(figsize=(max(5, len(questions)*0.8), 2.5))
        mode_labels = {"base": "Base", "predictive": "Predictive", "differentiable": "Differentiable"}
        ylabels = [mode_labels.get(m, m) for m in modes]

        cmap = plt.cm.YlGnBu
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(questions)))
        ax.set_xticklabels(questions)
        ax.set_yticks(np.arange(len(modes)))
        ax.set_yticklabels(ylabels)

        for i in range(len(modes)):
            for j in range(len(questions)):
                val = data[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

        plt.colorbar(im, ax=ax, label="ROUGE-L", shrink=0.8)
        wc = results[doc].get("word_count", "?")
        ax.set_title(f"Per-Question ROUGE-L: {doc.replace('_',' ').title()} ({wc} words)", fontweight="bold")
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f"fig5_heatmap_{doc}.png")
        fig.savefig(path)
        fig.savefig(path.replace(".png", ".pdf"))
        plt.close(fig)
        print(f"  Saved: {path}")


def fig6_timing_breakdown(results, plt):
    """Figure 6: Stacked bar — timing breakdown by pipeline stage."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    modes_all = []
    gist_times, lookup_times, other_times = [], [], []

    for doc in results:
        for mode in results[doc]["modes"]:
            md = results[doc]["modes"][mode]
            g = np.mean(md["gist_time"]) if md["gist_time"] else 0
            l = np.mean(md["lookup_time"]) if md["lookup_time"] else 0
            t = np.mean(md["time"]) if md["time"] else 0
            modes_all.append(f"{mode}\n({doc.split('_')[0]})")
            gist_times.append(g)
            lookup_times.append(l)
            other_times.append(max(0, t - g - l))

    x = np.arange(len(modes_all))
    ax.bar(x, gist_times, 0.6, label="Gisting", color="#4C72B0")
    ax.bar(x, lookup_times, 0.6, bottom=gist_times, label="Lookup", color="#DD8452")
    bottoms = [g + l for g, l in zip(gist_times, lookup_times)]
    ax.bar(x, other_times, 0.6, bottom=bottoms, label="Other (Pagination+Response)", color="#55A868")

    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(modes_all, fontsize=7)
    ax.legend(frameon=True, fancybox=False)
    ax.set_title("Pipeline Timing Breakdown", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig6_timing_breakdown.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


def fig7_improvement_bars(results, plt):
    """Figure 7: Improvement percentage over base ReadAgent."""
    metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "LR-1"]
    improved_modes = ["predictive", "differentiable"]
    mode_labels = ["Predictive Gisting", "Differentiable Retrieval"]
    colors = ["#DD8452", "#55A868"]

    # Average base values across docs
    base_vals = {m: [] for m in metrics}
    improved_vals = {mode: {m: [] for m in metrics} for mode in improved_modes}

    for doc in results:
        if "base" in results[doc]["modes"]:
            for m in metrics:
                base_vals[m].append(results[doc]["modes"]["base"]["averages"][m])
        for mode in improved_modes:
            if mode in results[doc]["modes"]:
                for m in metrics:
                    improved_vals[mode][m].append(results[doc]["modes"][mode]["averages"][m])

    base_avgs = {m: np.mean(base_vals[m]) for m in metrics}
    improvements = {}
    for mode in improved_modes:
        improvements[mode] = {}
        for m in metrics:
            imp_avg = np.mean(improved_vals[mode][m]) if improved_vals[mode][m] else 0
            if base_avgs[m] != 0:
                improvements[mode][m] = ((imp_avg - base_avgs[m]) / base_avgs[m]) * 100
            else:
                improvements[mode][m] = 0

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(metrics))
    width = 0.35
    for i, mode in enumerate(improved_modes):
        vals = [improvements[mode][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=mode_labels[i], color=colors[i], edgecolor="white")
        for bar, val in zip(bars, vals):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, y + (1 if y >= 0 else -3),
                   f"{val:+.1f}%", ha="center", va="bottom" if y >= 0 else "top", fontsize=7)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_ylabel("Improvement over Base (%)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=True, fancybox=False)
    ax.set_title("Improvement Over Base ReadAgent", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig7_improvement_bars.png")
    fig.savefig(path)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_latex_table(results):
    """Generate LaTeX table matching the base paper's format."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Evaluation Results Across Pipeline Modes}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & CR (\%) & \#LU & ROUGE-1 & ROUGE-L & LR-1 (\%) & LR-2 (\%) \\",
        r"\midrule",
    ]
    mode_labels = {"base": "ReadAgent-P (Base)", "predictive": "Predictive Gisting",
                   "differentiable": "Diff. Retrieval"}
    for doc in results:
        wc = results[doc].get("word_count", "?")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{doc.replace('_',' ').title()} ({wc} words)}}}} \\")
        for mode in results[doc]["modes"]:
            a = results[doc]["modes"][mode]["averages"]
            label = mode_labels.get(mode, mode)
            lines.append(
                f"  {label} & {a['CR']:.1f} & {a['Lookups']:.1f} & "
                f"{a['ROUGE-1']:.3f} & {a['ROUGE-L']:.3f} & "
                f"{a['LR-1']:.1f} & {a['LR-2']:.1f} \\\\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    table_str = "\n".join(lines)

    path = os.path.join(RESULTS_DIR, "results_table.tex")
    with open(path, "w") as f:
        f.write(table_str)
    print(f"  Saved: {path}")
    return table_str


def main():
    print("Loading results...")
    results = load_results()
    plt = setup_matplotlib()

    print("\nGenerating figures:")
    fig1_rouge_comparison(results, plt)
    fig2_llm_ratings(results, plt)
    fig3_cr_vs_quality(results, plt)
    fig4_token_efficiency(results, plt)
    fig5_per_question_heatmap(results, plt)
    fig6_timing_breakdown(results, plt)
    fig7_improvement_bars(results, plt)

    print("\nGenerating LaTeX table:")
    generate_latex_table(results)

    print(f"\nAll figures saved to: {FIGURES_DIR}/")
    print(f"LaTeX table saved to: {RESULTS_DIR}/results_table.tex")
    print("Done!")


if __name__ == "__main__":
    main()
