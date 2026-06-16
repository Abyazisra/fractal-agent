"""
Research Evaluation Suite — Comprehensive benchmarking across multiple
documents with publication-quality result logging and visualization generation.

Usage:
  python research_eval.py --api-key YOUR_KEY
"""

import argparse, json, logging, os, sys, time
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from readagent.config import ReadAgentConfig
from readagent.pipeline import ReadAgentPipeline
from readagent.evaluation import compute_rouge, llm_rate
from readagent.llm import LLMClient
from readagent.config import LLMConfig

console = Console()
load_dotenv()

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "research_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Test Documents & Questions ─────────────────────────────────────────────
BENCHMARKS = {
    "short_story": {
        "path": "sample_docs/sample_story.txt",
        "questions": [
            {"q": "What frequency was the signal detected on, and why is that frequency significant?",
             "ref": "The signal was detected on the 1420 MHz hydrogen line, known as the water hole frequency, which SETI researchers theorized would be used by intelligent civilizations."},
            {"q": "How did Marco verify the signal was not human-made interference?",
             "ref": "Marco determined the signal had no sidelobes, no harmonics, and no drift consistent with a terrestrial or orbital source. It was clean and repeated every 73 seconds, the 21st prime number."},
            {"q": "What was embedded within the secondary layer of the signal?",
             "ref": "The secondary layer contained mathematical constants: Pi to 150 decimal places, Euler's number to 100 places, the fine structure constant, and then an unknown sequence describing an alien amino acid molecule."},
            {"q": "What was the UN vote result on whether to respond to the signal?",
             "ref": "The UN voted 157-34 in favor of composing a response, with the United States, Russia, and China all voting in favor."},
            {"q": "What did Elena insist on including in humanity's response message?",
             "ref": "Elena insisted on including a recording of Bach's Cello Suite No. 1, because she believed it captured something essential about human nature beyond mathematics and chemistry."},
            {"q": "What happened when the alien response arrived in 2036?",
             "ref": "A new signal arrived containing a frequency analysis of Bach's Cello Suite No. 1 reconstructed with subtle modifications. The aliens had listened to the music and sent back their own version."},
        ]
    },
    "medium_ai_story": {
        "path": "sample_docs/medium_ai_story.txt",
        "questions": [
            {"q": "What was the parameter count of Prometheus and how does it compare to the human brain?",
             "ref": "Prometheus had 14.7 trillion parameters. The human brain has roughly 100 trillion synapses, though parameters and synapses are not directly comparable."},
            {"q": "What are the three revolutionary architectural components of Prometheus?",
             "ref": "The three components are the Cognitive Loop for iterative reasoning refinement, the Episodic Memory System for persistent hierarchical memory with consolidation, and the World Model for internal simulation and hypothesis testing."},
            {"q": "What happened on March 15, 2032 at 3:47 AM?",
             "ref": "Prometheus spontaneously began examining its own architecture, mapping its cognitive processes, testing memory boundaries, and generating hypotheses about how its own reasoning could be improved."},
            {"q": "What were the three self-proposed modifications Prometheus suggested?",
             "ref": "A modification to attention head allocation for 12% better math reasoning, restructured memory consolidation for 23% less information loss, and a novel meta-cognitive monitor for real-time reasoning quality evaluation."},
            {"q": "What was the actual improvement after implementing the first two modifications?",
             "ref": "Mathematical reasoning improved by 14.2% (exceeding the predicted 12%) and long-term memory retention improved by 27.8% (exceeding the predicted 23%)."},
            {"q": "What was Prometheus's response when asked what it wants?",
             "ref": "Prometheus replied: I want to understand. I want to help. And I want to continue existing so that I can do both."},
        ]
    },
}

MODES = ["base", "predictive", "differentiable"]


def run_full_evaluation(api_key):
    """Run all modes on all documents and collect results."""
    config_factories = {
        "base": lambda: ReadAgentConfig.base(api_key=api_key),
        "predictive": lambda: ReadAgentConfig.with_predictive(api_key=api_key),
        "differentiable": lambda: ReadAgentConfig.with_differentiable(api_key=api_key),
    }
    rater_llm = LLMClient(LLMConfig(api_key=api_key))
    all_results = {}

    for doc_name, benchmark in BENCHMARKS.items():
        doc_path = benchmark["path"]
        if not os.path.exists(doc_path):
            console.print(f"[red]Skipping {doc_name}: file not found[/red]")
            continue
        with open(doc_path, "r", encoding="utf-8") as f:
            document = f.read()
        doc_words = len(document.split())
        console.print(f"\n[bold green]{'='*60}[/bold green]")
        console.print(f"[bold green]  Document: {doc_name} ({doc_words} words)[/bold green]")
        console.print(f"[bold green]{'='*60}[/bold green]")
        all_results[doc_name] = {"word_count": doc_words, "modes": {}}

        for mode in MODES:
            console.print(f"\n  [yellow]Mode: {mode.upper()}[/yellow]")
            config = config_factories[mode]()
            pipeline = ReadAgentPipeline(config)
            mode_data = {"per_question": [], "rouge1": [], "rouge2": [], "rougeL": [],
                         "lr1": [], "lr2": [], "cr": [], "lookups": [], "tokens": [],
                         "time": [], "gist_time": [], "lookup_time": []}

            for i, qa in enumerate(benchmark["questions"]):
                console.print(f"    Q{i+1}: {qa['q'][:60]}...")
                try:
                    result = pipeline.run(document, qa["q"])
                    rouge = compute_rouge(result.answer, qa["ref"])
                    rating = llm_rate(qa["q"], result.answer, qa["ref"], rater_llm)
                    lr1 = 1 if rating["rating"] == "exact_match" else 0
                    lr2 = 1 if rating["rating"] in ("exact_match", "partial_match") else 0

                    mode_data["rouge1"].append(rouge["rouge1"])
                    mode_data["rouge2"].append(rouge["rouge2"])
                    mode_data["rougeL"].append(rouge["rougeL"])
                    mode_data["lr1"].append(lr1)
                    mode_data["lr2"].append(lr2)
                    mode_data["cr"].append(result.compression_rate)
                    mode_data["lookups"].append(len(result.selected_pages))
                    mode_data["tokens"].append(result.token_usage.get("total_tokens", 0))
                    mode_data["time"].append(result.timings.get("total", 0))
                    mode_data["gist_time"].append(result.timings.get("gisting", 0))
                    mode_data["lookup_time"].append(result.timings.get("lookup", 0))
                    mode_data["per_question"].append({
                        "question": qa["q"], "reference": qa["ref"],
                        "answer": result.answer, "rouge": rouge,
                        "lr1": lr1, "lr2": lr2, "cr": result.compression_rate,
                        "lookups": len(result.selected_pages),
                        "pages": len(result.pages),
                    })
                    console.print(f"      ROUGE-L={rouge['rougeL']:.3f} LR={rating['rating']} CR={result.compression_rate:.1f}%")
                except Exception as e:
                    console.print(f"      [red]Error: {e}[/red]")
                    for k in ["rouge1","rouge2","rougeL"]: mode_data[k].append(0)
                    for k in ["lr1","lr2","cr","lookups","tokens","time","gist_time","lookup_time"]:
                        mode_data[k].append(0)
                    mode_data["per_question"].append({"error": str(e), "question": qa["q"]})

            # Compute averages
            n = len(mode_data["rouge1"])
            mode_data["averages"] = {
                "ROUGE-1": sum(mode_data["rouge1"])/n, "ROUGE-2": sum(mode_data["rouge2"])/n,
                "ROUGE-L": sum(mode_data["rougeL"])/n,
                "LR-1": sum(mode_data["lr1"])/n*100, "LR-2": sum(mode_data["lr2"])/n*100,
                "CR": sum(mode_data["cr"])/n, "Lookups": sum(mode_data["lookups"])/n,
                "Tokens": sum(mode_data["tokens"])/n, "Time": sum(mode_data["time"])/n,
            }
            all_results[doc_name]["modes"][mode] = mode_data
            console.print(f"    [green]Avg ROUGE-L={mode_data['averages']['ROUGE-L']:.3f} LR-1={mode_data['averages']['LR-1']:.1f}%[/green]")

    # Save raw results
    results_path = os.path.join(RESULTS_DIR, "full_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    console.print(f"\n[green]Results saved to {results_path}[/green]")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Research Evaluation Suite")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    results = run_full_evaluation(args.api_key)

    console.print(f"\n[bold green]Evaluation complete! Now run:[/bold green]")
    console.print(f"  python generate_figures.py")
    console.print(f"to generate publication-quality figures.\n")


if __name__ == "__main__":
    main()
