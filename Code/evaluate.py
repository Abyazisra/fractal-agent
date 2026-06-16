"""
Evaluation Benchmark — Runs all pipeline modes on a set of questions
with reference answers and computes the base paper's exact metrics:
  - Compression Rate (CR)
  - ROUGE-1, ROUGE-2, ROUGE-L
  - LLM Rating-1 (strict), LLM Rating-2 (permissive)
  - Number of Lookups (# LU)
  - Token usage and timing

Usage:
  python evaluate.py --api-key YOUR_KEY
  python evaluate.py --api-key YOUR_KEY --modes base predictive differentiable
"""

import argparse
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from readagent.config import ReadAgentConfig
from readagent.pipeline import ReadAgentPipeline
from readagent.evaluation import compute_rouge, llm_rate
from readagent.llm import LLMClient
from readagent.config import LLMConfig

console = Console()
load_dotenv()

# ─── Benchmark Questions with Reference Answers ────────────────────────────
SAMPLE_DOC = os.path.join(os.path.dirname(__file__), "sample_docs", "sample_story.txt")

BENCHMARK_QA = [
    {
        "question": "What frequency was the signal detected on, and why is that frequency significant?",
        "reference": "The signal was detected on the 1420 MHz hydrogen line, known as the water hole frequency, which SETI researchers theorized would be used by intelligent civilizations.",
    },
    {
        "question": "How did Marco verify the signal was not human-made interference?",
        "reference": "Marco determined the signal had no sidelobes, no harmonics, and no drift consistent with a terrestrial or orbital source. It was clean and repeated every 73 seconds, the 21st prime number.",
    },
    {
        "question": "What was embedded within the secondary layer of the signal?",
        "reference": "The secondary layer contained mathematical constants: Pi to 150 decimal places, Euler's number to 100 places, the fine structure constant, and then an unknown sequence that turned out to be a description of an alien amino acid molecule.",
    },
    {
        "question": "What was the UN vote result on whether to respond to the signal?",
        "reference": "The UN voted 157-34 in favor of composing a response, with the United States, Russia, and China all voting in favor.",
    },
    {
        "question": "What did Elena insist on including in humanity's response message, and why?",
        "reference": "Elena insisted on including a recording of Bach's Cello Suite No. 1, because she believed it captured something essential about human nature beyond mathematics and chemistry.",
    },
    {
        "question": "What happened in the epilogue when the alien response arrived?",
        "reference": "On November 14, 2036, a new signal arrived containing a frequency analysis of Bach's Cello Suite No. 1 reconstructed with subtle modifications. The aliens had listened to the music and sent back their own version.",
    },
]


def run_evaluation(api_key, modes, verbose=False):
    """Run all modes on all questions and compute metrics."""
    # Load document
    with open(SAMPLE_DOC, "r", encoding="utf-8") as f:
        document = f.read()

    # LLM client for rating
    rater_llm = LLMClient(LLMConfig(api_key=api_key))

    config_factories = {
        "base": lambda: ReadAgentConfig.base(api_key=api_key),
        "fractal": lambda: ReadAgentConfig.with_fractal(api_key=api_key),
        "predictive": lambda: ReadAgentConfig.with_predictive(api_key=api_key),
        "differentiable": lambda: ReadAgentConfig.with_differentiable(api_key=api_key),
        "full": lambda: ReadAgentConfig.full(api_key=api_key),
    }

    all_results = {}

    for mode in modes:
        console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
        console.print(f"[bold yellow]  Evaluating: {mode.upper()}[/bold yellow]")
        console.print(f"[bold yellow]{'='*60}[/bold yellow]")

        config = config_factories[mode]()
        pipeline = ReadAgentPipeline(config)

        mode_results = {
            "answers": [],
            "rouge1": [], "rouge2": [], "rougeL": [],
            "lr1_ratings": [], "lr2_ratings": [],
            "compression_rates": [],
            "num_lookups": [],
            "total_tokens": [],
            "timings": [],
        }

        for i, qa in enumerate(BENCHMARK_QA):
            console.print(f"\n  [cyan]Q{i+1}:[/cyan] {qa['question'][:80]}...")

            try:
                result = pipeline.run(document, qa["question"])
                answer = result.answer

                # ROUGE scores
                rouge = compute_rouge(answer, qa["reference"])
                mode_results["rouge1"].append(rouge["rouge1"])
                mode_results["rouge2"].append(rouge["rouge2"])
                mode_results["rougeL"].append(rouge["rougeL"])

                # LLM Ratings
                rating = llm_rate(qa["question"], answer, qa["reference"], rater_llm)
                mode_results["lr1_ratings"].append(1 if rating["rating"] == "exact_match" else 0)
                mode_results["lr2_ratings"].append(
                    1 if rating["rating"] in ("exact_match", "partial_match") else 0
                )

                # Other metrics
                mode_results["compression_rates"].append(result.compression_rate)
                mode_results["num_lookups"].append(len(result.selected_pages))
                mode_results["total_tokens"].append(result.token_usage.get("total_tokens", 0))
                mode_results["timings"].append(result.timings.get("total", 0))
                mode_results["answers"].append(answer)

                console.print(f"  [green]A:[/green] {answer[:100]}...")
                console.print(
                    f"  ROUGE-L={rouge['rougeL']:.3f} | "
                    f"Rating={rating['rating']} | "
                    f"CR={result.compression_rate:.1f}% | "
                    f"LU={len(result.selected_pages)}"
                )

            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                # Append zeros for failed questions
                for key in ["rouge1", "rouge2", "rougeL"]:
                    mode_results[key].append(0.0)
                mode_results["lr1_ratings"].append(0)
                mode_results["lr2_ratings"].append(0)
                mode_results["compression_rates"].append(0)
                mode_results["num_lookups"].append(0)
                mode_results["total_tokens"].append(0)
                mode_results["timings"].append(0)
                mode_results["answers"].append(f"ERROR: {e}")

        all_results[mode] = mode_results

    return all_results


def compute_averages(mode_results):
    """Compute average metrics for a mode."""
    n = len(mode_results["rouge1"])
    if n == 0:
        return {}
    return {
        "ROUGE-1": sum(mode_results["rouge1"]) / n,
        "ROUGE-2": sum(mode_results["rouge2"]) / n,
        "ROUGE-L": sum(mode_results["rougeL"]) / n,
        "LR-1": sum(mode_results["lr1_ratings"]) / n * 100,
        "LR-2": sum(mode_results["lr2_ratings"]) / n * 100,
        "CR": sum(mode_results["compression_rates"]) / n,
        "# LU": sum(mode_results["num_lookups"]) / n,
        "Tokens": sum(mode_results["total_tokens"]) / n,
        "Time (s)": sum(mode_results["timings"]) / n,
    }


def display_comparison(all_results):
    """Display comparison table matching the base paper's format."""
    console.print(f"\n[bold green]{'='*70}[/bold green]")
    console.print(f"[bold green]  EVALUATION RESULTS — Base Paper Metrics[/bold green]")
    console.print(f"[bold green]{'='*70}[/bold green]\n")

    table = Table(title="ReadAgent Evaluation (6 Questions)", show_lines=True)
    table.add_column("Metric", style="bold", min_width=12)
    for mode in all_results:
        table.add_column(mode.upper(), justify="right", min_width=14)

    # Compute averages
    averages = {mode: compute_averages(results) for mode, results in all_results.items()}

    metrics = ["CR", "# LU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "LR-1", "LR-2", "Tokens", "Time (s)"]
    formats = {
        "CR": "{:.1f}%", "# LU": "{:.1f}", "ROUGE-1": "{:.3f}", "ROUGE-2": "{:.3f}",
        "ROUGE-L": "{:.3f}", "LR-1": "{:.1f}%", "LR-2": "{:.1f}%",
        "Tokens": "{:.0f}", "Time (s)": "{:.1f}",
    }

    for metric in metrics:
        row = [metric]
        for mode in all_results:
            val = averages[mode].get(metric, 0)
            row.append(formats[metric].format(val))
        table.add_row(*row)

    console.print(table)

    # Paper comparison reference
    console.print(Panel(
        "[bold]Base Paper Reference (QuALITY, PaLM 2-L):[/bold]\n"
        "  GistMem:           CR=85.5%  Accuracy=77.5%\n"
        "  ReadAgent-P 1-2pg: CR=72.2%  Accuracy=86.2%  #LU=1.6\n"
        "  ReadAgent-P 1-5pg: CR=66.5%  Accuracy=86.8%  #LU=2.3\n"
        "  ReadAgent-S 1-6pg: CR=58.5%  Accuracy=87.2%  #LU=3.2\n"
        "  Full Raw Content:  CR=0%     Accuracy=85.8%\n\n"
        "[bold]Base Paper Reference (QMSum, PaLM 2-L):[/bold]\n"
        "  GistMem:           CR=83.1%  LR-1=40.2%  LR-2=89.8%  ROUGE-L=20.2\n"
        "  ReadAgent-P 1-4pg: CR=73.5%  LR-1=40.0%  LR-2=90.6%  ROUGE-L=20.3\n"
        "  ReadAgent-S 1-6pg: CR=70.3%  LR-1=46.6%  LR-2=91.5%  ROUGE-L=21.2",
        title="[cyan]Paper Baselines for Comparison[/cyan]",
        border_style="cyan",
    ))

    # Save results to JSON
    output = {
        mode: {
            "averages": averages[mode],
            "per_question": {
                f"Q{i+1}": {
                    "question": BENCHMARK_QA[i]["question"],
                    "answer": results["answers"][i],
                    "reference": BENCHMARK_QA[i]["reference"],
                    "rouge1": results["rouge1"][i],
                    "rouge2": results["rouge2"][i],
                    "rougeL": results["rougeL"][i],
                    "lr1": results["lr1_ratings"][i],
                    "lr2": results["lr2_ratings"][i],
                }
                for i in range(len(results["answers"]))
            }
        }
        for mode, results in all_results.items()
    }

    output_path = "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    console.print(f"\n[green]Detailed results saved to {output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="ReadAgent Evaluation Benchmark")
    parser.add_argument("--api-key", required=True, help="Mistral API key")
    parser.add_argument("--modes", nargs="+",
                        default=["base", "predictive", "differentiable"],
                        choices=["base", "fractal", "predictive", "differentiable", "full"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console.print(Panel(
        f"[bold]Document:[/bold] {SAMPLE_DOC}\n"
        f"[bold]Questions:[/bold] {len(BENCHMARK_QA)}\n"
        f"[bold]Modes:[/bold] {', '.join(args.modes)}\n"
        f"[bold]Metrics:[/bold] CR, ROUGE-1/2/L, LLM-Rating-1/2, #LU, Tokens",
        title="[bold green]ReadAgent Evaluation Benchmark[/bold green]",
        border_style="green",
    ))

    all_results = run_evaluation(args.api_key, args.modes, args.verbose)
    display_comparison(all_results)
    console.print("\n[bold green]Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
