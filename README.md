# Beyond Episodic Memory: FractalAgent for Long-Context LLM Reading

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Mistral API](https://img.shields.io/badge/LLM-Mistral%20API-FF7000?style=flat-square)](https://mistral.ai)
[![Sentence Transformers](https://img.shields.io/badge/Embeddings-Sentence--Transformers-F7931E?style=flat-square)](https://sbert.net)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![FAST NUCES](https://img.shields.io/badge/Institution-FAST%20NUCES-blue?style=flat-square)](https://nu.edu.pk)

> **Research Paper:** *Beyond Episodic Memory: Addressing the Contextual and Computational Limitations of Human-Inspired Reading Agents*
> Muhammad Qasim · Ayaan Khan · **Abyaz Israr** — FAST NUCES, Islamabad

---

## Problem

Large Language Models fail on long documents. The quadratic complexity of self-attention creates a hard context window limit, and even within that limit LLMs suffer from the **"lost in the middle"** problem — performance degrades when relevant information appears far from the prompt boundaries.

[ReadAgent](https://arxiv.org/abs/2402.09727) addressed this with human-inspired episodic gist memory. But it has three critical limitations:

| Limitation | Impact |
|---|---|
| Flat gist memory hits context wall | Hard document length ceiling |
| Unconditional gisting loses key facts | Hallucinations on specific queries |
| LLM-based lookup per query | High latency — full inference every time |

---

## Our Solution — Three Innovations

### 1. FractalAgent — Recursive Context Scaling
When gist memory exceeds the context limit, FractalAgent recursively compresses gists into **meta-gists**, forming a hierarchical tree. Lookup navigates top-down through the tree — enabling **infinite context scaling** regardless of document length.

### 2. Predictive Task-Driven Gisting
Before compressing, a forecasting agent reads the first few pages and predicts downstream questions. Gisting then **preserves names, dates, numbers and facts** most likely to be queried — reducing hallucination on specific lookups.

### 3. Differentiable Gist Retrieval
Replaces LLM-based page lookup with **local cosine similarity over sentence-transformer embeddings**. Retrieval runs on GPU via matrix multiplication — no LLM inference call needed per query.

---

## Results

| Metric | Base ReadAgent | Predictive Gisting | Differentiable Retrieval |
|---|---|---|---|
| ROUGE-L (short story) | 0.329 | **0.484** (+47.1%) | 0.406 |
| ROUGE-L (medium story) | 0.573 | **0.594** | 0.533 |
| LR-2 Permissive | 83.3% | 83.3% | **100.0%** |
| Avg lookup time | baseline | baseline | **0.13s (84% faster)** |
| FractalAgent scaling | ❌ fails at context wall | ❌ fails at context wall | ✅ 35,491 words (H.G. Wells) |

**Key results:**
- **19.1% average ROUGE-L improvement** with Predictive Gisting
- **84% reduction in retrieval latency** with Differentiable Retrieval
- FractalAgent successfully indexed and queried *The Time Machine* (35,491 words) — 79 pages → 16 meta-gists → 1,296 words of root memory, answering complex plot questions correctly

---

## System Architecture

```
Raw Long Document
        │
        ▼
1. Autonomous Pagination     (LLM finds natural episode breaks)
        │
        ▼
2. Episodic Gisting    ◄───  Improvement 2: Predictive Task-Driven Gisting
        │
        ├── gist size > threshold ──► Improvement 1: FractalAgent (recursive meta-gists)
        │
        ▼
3. Page Retrieval      ◄───  Improvement 3: Differentiable Retrieval (cosine similarity)
        │
        ▼
4. Final Response Synthesis
        │
        ▼
   Accurate Answer
```

---

## Repository Structure

```
AgenticProject/
│
├── Code/
│   ├── readagent/                      ← core pipeline package
│   │   ├── pipeline.py                 ← main orchestrator (all modes)
│   │   ├── pagination.py               ← LLM-guided episode splitting
│   │   ├── gisting.py                  ← unconditional gist compression
│   │   ├── lookup.py                   ← LLM-based page retrieval
│   │   ├── fractal_agent.py            ← recursive tree construction + lookup
│   │   ├── predictive_gisting.py       ← task forecasting + task-aware gisting
│   │   └── differentiable_retrieval.py ← embedding cosine similarity retrieval
│   │
│   ├── research_results/               ← raw experiment outputs
│   ├── sample_docs/                    ← test documents used in evaluation
│   │
│   ├── main.py                         ← main entry point
│   ├── demo.py                         ← quick demo script
│   ├── evaluate.py                     ← evaluation runner
│   ├── research_eval.py                ← full benchmarking + metrics
│   ├── generate_figures.py             ← ROUGE plots, heatmaps, timing charts
│   ├── implementation_plan.md          ← design notes and architecture decisions
│   ├── evaluation_results.json         ← full benchmark results
│   ├── base_text.txt                   ← base document used in experiments
│   ├── improvement_text.txt            ← improved system comparison text
│   └── requirements.txt               ← dependencies
│
├── AgenticResearchPaper.pdf            ← full research paper
├── Poster Presentation.pptx           ← conference poster
└── README.md
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/Abyazisra/fractal-agent.git
cd fractal-agent/Code
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
Create a `.env` file in the `Code/` folder:
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 4. Run the demo
```bash
python demo.py
```

### 5. Run with a specific mode
```bash
python main.py --mode base
python main.py --mode predictive
python main.py --mode differentiable
python main.py --mode fractal
```

### 6. Run the full benchmark
```bash
python research_eval.py
```

### 7. Reproduce the paper figures
```bash
python generate_figures.py
```

---

## Hardware Used

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 3060 Ti (8GB VRAM) |
| CPU | AMD Ryzen 5 5600 |
| RAM | 16GB DDR4 |
| LLM API | Mistral (`mistral-small-latest`) |
| Embeddings | `sentence-transformers` (local GPU) |

---

## FractalAgent Scalability Test

We tested on *The Time Machine* by H.G. Wells (35,491 words):

- Pagination agent generated **79 pages**
- Total gist memory: ~9,000 words → **exceeded 6,000 token threshold**
- FractalAgent triggered: 79 gists → **16 meta-gists** → **1,296 words** of root memory
- Query: *"What happens when the Time Traveller meets the Eloi and Morlocks?"*
- Result: Correct, detailed answer — traversing branches 4–12 of the tree
- Cost: 175 LLM calls · 188,854 tokens · 5.2 minutes initial indexing

---

## Citation

If you use this work, please cite:

```bibtex
@article{qasim2025fractalagent,
  title={Beyond Episodic Memory: Addressing the Contextual and Computational
         Limitations of Human-Inspired Reading Agents},
  author={Qasim, Muhammad and Khan, Ayaan and Israr, Abyaz},
  institution={FAST NUCES, Islamabad, Pakistan},
  year={2025}
}
```

---

## Authors

| Name | Student ID | Institution |
|---|---|---|
| Muhammad Qasim | I221994 | FAST NUCES, Islamabad |
| Ayaan Khan | I222066 | FAST NUCES, Islamabad |
| Abyaz Israr | I222056 | FAST NUCES, Islamabad |

---

## References

This work builds on:

- [ReadAgent](https://arxiv.org/abs/2402.09727) — Lee et al., 2024
- [MemWalker](https://arxiv.org/abs/2310.05029) — Chen et al., 2023
- [MemGPT](https://arxiv.org/abs/2310.08560) — Packer et al., 2023
- [ReAct](https://arxiv.org/abs/2210.03629) — Yao et al., 2023
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — Liu et al., 2024
- [LongLoRA](https://arxiv.org/abs/2309.12307) — Chen et al., 2023
- [RingAttention](https://arxiv.org/abs/2310.01889) — Liu et al., 2023

---

*Part of the Data Science and Artificial Intelligence programme at FAST NUCES, Islamabad.*
