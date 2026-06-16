"""
ReadAgent CLI — Main entry point for running the pipeline.

Usage:
    python main.py --mode base --document doc.txt --question "What happened?"
    python main.py --mode full --document doc.txt --question "Who is the protagonist?"
    python main.py --mode fractal --document doc.txt --question "..."
    python main.py --mode predictive --document doc.txt --question "..."
    python main.py --mode differentiable --document doc.txt --question "..."
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from readagent.config import ReadAgentConfig
from readagent.pipeline import ReadAgentPipeline

console = Console()
load_dotenv()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def create_config(mode: str, api_key: str, **kwargs) -> ReadAgentConfig:
    """Create configuration based on mode."""
    config_map = {
        "base": ReadAgentConfig.base,
        "fractal": ReadAgentConfig.with_fractal,
        "predictive": ReadAgentConfig.with_predictive,
        "differentiable": ReadAgentConfig.with_differentiable,
        "full": ReadAgentConfig.full,
    }
    factory = config_map.get(mode, ReadAgentConfig.base)
    config = factory(api_key=api_key)

    # Apply any overrides
    if kwargs.get("max_lookups"):
        config.lookup.max_lookups = kwargs["max_lookups"]
    if kwargs.get("strategy"):
        config.lookup.strategy = kwargs["strategy"]
    if kwargs.get("min_words"):
        config.pagination.min_words = kwargs["min_words"]
    if kwargs.get("max_words"):
        config.pagination.max_words = kwargs["max_words"]

    return config


def display_result(result, mode: str):
    """Display pipeline results using rich formatting."""
    console.print()
    console.print(Panel(f"[bold green]ReadAgent Results[/bold green] (mode: {mode})", expand=False))

    # Answer
    console.print(Panel(result.answer, title="[bold cyan]Answer[/bold cyan]", border_style="cyan"))

    # Stats table
    stats = Table(title="Pipeline Statistics")
    stats.add_column("Metric", style="bold")
    stats.add_column("Value", justify="right")

    stats.add_row("Mode", result.mode)
    stats.add_row("Pages Created", str(len(result.pages)))
    stats.add_row("Compression Rate", f"{result.compression_rate:.1f}%")
    stats.add_row("Pages Looked Up", str(result.selected_pages))
    stats.add_row("Total Time", f"{result.timings.get('total', 0):.1f}s")
    stats.add_row("  Pagination", f"{result.timings.get('pagination', 0):.1f}s")
    stats.add_row("  Gisting", f"{result.timings.get('gisting', 0):.1f}s")
    stats.add_row("  Lookup", f"{result.timings.get('lookup', 0):.1f}s")
    stats.add_row("  Response", f"{result.timings.get('response', 0):.1f}s")
    stats.add_row("LLM Calls", str(result.token_usage.get("call_count", 0)))
    stats.add_row("Total Tokens", str(result.token_usage.get("total_tokens", 0)))

    console.print(stats)

    # Metadata
    if result.metadata:
        meta_table = Table(title="Extension Metadata")
        meta_table.add_column("Key", style="bold")
        meta_table.add_column("Value")
        for k, v in result.metadata.items():
            if k == "importance":
                high = sum(1 for m in v if m["importance"] == "HIGH")
                med = sum(1 for m in v if m["importance"] == "MEDIUM")
                low = sum(1 for m in v if m["importance"] == "LOW")
                meta_table.add_row("Page Importance", f"HIGH={high}, MEDIUM={med}, LOW={low}")
            else:
                meta_table.add_row(str(k), str(v))
        console.print(meta_table)


def main():
    parser = argparse.ArgumentParser(description="ReadAgent: Long-Context Reading Agent")
    parser.add_argument("--mode", choices=["base", "fractal", "predictive", "differentiable", "full"],
                        default="base", help="Pipeline mode")
    parser.add_argument("--document", required=True, help="Path to the document file")
    parser.add_argument("--question", required=True, help="Question to answer")
    parser.add_argument("--options", default=None, help="MCQ options (optional)")
    parser.add_argument("--api-key", default=None, help="Mistral API key (or set MISTRAL_API_KEY env var)")
    parser.add_argument("--strategy", choices=["parallel", "sequential"], default="parallel")
    parser.add_argument("--max-lookups", type=int, default=5)
    parser.add_argument("--min-words", type=int, default=280)
    parser.add_argument("--max-words", type=int, default=600)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Get API key
    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        console.print("[bold red]Error: No API key provided.[/bold red]")
        console.print("Set MISTRAL_API_KEY environment variable or use --api-key flag.")
        sys.exit(1)

    # Read document
    if not os.path.exists(args.document):
        console.print(f"[bold red]Error: Document not found: {args.document}[/bold red]")
        sys.exit(1)

    with open(args.document, "r", encoding="utf-8") as f:
        document = f.read()

    console.print(f"[bold]Document loaded:[/bold] {len(document.split())} words")
    console.print(f"[bold]Question:[/bold] {args.question}")
    console.print(f"[bold]Mode:[/bold] {args.mode}")
    console.print()

    # Create config and run
    config = create_config(
        args.mode, api_key,
        max_lookups=args.max_lookups,
        strategy=args.strategy,
        min_words=args.min_words,
        max_words=args.max_words,
    )
    pipeline = ReadAgentPipeline(config)
    result = pipeline.run(document, args.question, args.options)

    display_result(result, args.mode)


if __name__ == "__main__":
    main()
