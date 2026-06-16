"""
Interactive Demo — Runs all modes on a sample document and compares results.

Usage: python demo.py

Loads MISTRAL_API_KEY from .env or the current environment.
"""

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from readagent.config import ReadAgentConfig
from readagent.pipeline import ReadAgentPipeline, PipelineResult

console = Console()
load_dotenv()

SAMPLE_DOC_PATH = os.path.join(os.path.dirname(__file__), "sample_docs", "sample_story.txt")


def run_mode(mode_name, config, document, question, options=None):
    """Run a single mode and return the result."""
    console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
    console.print(f"[bold yellow]  Running: {mode_name}[/bold yellow]")
    console.print(f"[bold yellow]{'='*60}[/bold yellow]")

    pipeline = ReadAgentPipeline(config)
    result = pipeline.run(document, question, options)

    console.print(Panel(
        result.answer,
        title=f"[bold cyan]{mode_name} Answer[/bold cyan]",
        border_style="cyan",
    ))

    return result


def compare_results(results: dict):
    """Display a comparison table of all mode results."""
    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green]  COMPARISON TABLE[/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]\n")

    table = Table(title="Mode Comparison", show_lines=True)
    table.add_column("Metric", style="bold", min_width=20)

    for mode_name in results:
        table.add_column(mode_name, min_width=15)

    # Rows
    metrics = [
        ("Pages", lambda r: str(len(r.pages))),
        ("Compression Rate", lambda r: f"{r.compression_rate:.1f}%"),
        ("Pages Looked Up", lambda r: str(r.selected_pages)),
        ("Retrieval Method", lambda r: r.metadata.get("retrieval_method", "N/A")),
        ("LLM Calls", lambda r: str(r.token_usage.get("call_count", 0))),
        ("Total Tokens", lambda r: str(r.token_usage.get("total_tokens", 0))),
        ("Pagination Time", lambda r: f"{r.timings.get('pagination', 0):.1f}s"),
        ("Gisting Time", lambda r: f"{r.timings.get('gisting', 0):.1f}s"),
        ("Lookup Time", lambda r: f"{r.timings.get('lookup', 0):.1f}s"),
        ("Response Time", lambda r: f"{r.timings.get('response', 0):.1f}s"),
        ("Total Time", lambda r: f"{r.timings.get('total', 0):.1f}s"),
    ]

    for metric_name, extractor in metrics:
        row = [metric_name]
        for mode_name, result in results.items():
            row.append(extractor(result))
        table.add_row(*row)

    # Add answers row (truncated)
    answer_row = ["Answer (truncated)"]
    for mode_name, result in results.items():
        answer_row.append(result.answer[:80] + "..." if len(result.answer) > 80 else result.answer)
    table.add_row(*answer_row)

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="ReadAgent Interactive Demo")
    parser.add_argument("--api-key", default=None, help="Optional Mistral API key override; defaults to MISTRAL_API_KEY from .env")
    parser.add_argument("--document", default=None, help="Document path (uses sample if not provided)")
    parser.add_argument("--question", default=None, help="Question (uses sample if not provided)")
    parser.add_argument("--modes", nargs="+",
                        default=["base", "predictive", "fractal", "differentiable"],
                        choices=["base", "fractal", "predictive", "differentiable", "full"],
                        help="Modes to compare")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Load document
    doc_path = args.document or SAMPLE_DOC_PATH
    if not os.path.exists(doc_path):
        console.print(f"[bold red]Document not found: {doc_path}[/bold red]")
        sys.exit(1)

    with open(doc_path, "r", encoding="utf-8") as f:
        document = f.read()

    # Default question
    question = args.question or "What are the main events that happen in this story, and what is the protagonist's key motivation?"

    console.print(Panel(
        f"[bold]Document:[/bold] {doc_path}\n"
        f"[bold]Words:[/bold] {len(document.split())}\n"
        f"[bold]Question:[/bold] {question}\n"
        f"[bold]Modes:[/bold] {', '.join(args.modes)}",
        title="[bold green]ReadAgent Demo[/bold green]",
        border_style="green",
    ))

    # Run each mode
    results = {}
    config_factories = {
        "base": lambda: ReadAgentConfig.base(api_key=args.api_key),
        "fractal": lambda: ReadAgentConfig.with_fractal(api_key=args.api_key),
        "predictive": lambda: ReadAgentConfig.with_predictive(api_key=args.api_key),
        "differentiable": lambda: ReadAgentConfig.with_differentiable(api_key=args.api_key),
        "full": lambda: ReadAgentConfig.full(api_key=args.api_key),
    }

    for mode in args.modes:
        try:
            config = config_factories[mode]()
            result = run_mode(mode.upper(), config, document, question)
            results[mode] = result
        except Exception as e:
            console.print(f"[bold red]Error in {mode} mode: {e}[/bold red]")
            import traceback
            traceback.print_exc()

    # Compare
    if len(results) > 1:
        compare_results(results)

    console.print("\n[bold green]Demo complete![/bold green]")


if __name__ == "__main__":
    main()
