"""
ReadAgent Pipeline — Orchestrates the full workflow.

Supports four modes:
  - base: Standard ReadAgent (pagination → gisting → lookup → response)
  - fractal: ReadAgent + FractalAgent for higher context scaling
  - predictive: ReadAgent + Predictive Task-Driven Gisting
  - differentiable: ReadAgent + Differentiable Gist Retrieval
  - full: All three improvements combined
"""

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .config import ReadAgentConfig
from .llm import LLMClient
from .pagination import paginate_document
from .gisting import gist_pages
from .lookup import parallel_lookup, sequential_lookup, generate_response
from .fractal_agent import build_fractal_tree, fractal_lookup
from .predictive_gisting import predictive_gist_pages
from .utils import Page, GistMemory, count_words

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Stores all outputs from a pipeline run."""
    answer: str = ""
    pages: list = field(default_factory=list)
    gist_memory: Optional[GistMemory] = None
    selected_pages: list = field(default_factory=list)
    expanded_memory: str = ""
    compression_rate: float = 0.0
    mode: str = "base"
    timings: dict = field(default_factory=dict)
    token_usage: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class ReadAgentPipeline:
    """Main pipeline orchestrator for ReadAgent and its extensions."""

    def __init__(self, config: ReadAgentConfig):
        self.config = config
        self.llm = LLMClient(config.llm)
        self._diff_retriever = None

    def _get_diff_retriever(self):
        """Lazy-initialize the differentiable retriever."""
        if self._diff_retriever is None:
            from .differentiable_retrieval import DifferentiableRetriever
            self._diff_retriever = DifferentiableRetriever(
                embedding_model_name=self.config.differentiable.embedding_model,
                hidden_dim=self.config.differentiable.adapter_hidden_dim,
                use_cross_attention=self.config.differentiable.use_cross_attention,
            )
        return self._diff_retriever

    def run(
        self,
        document: str,
        question: str,
        options: str = None,
    ) -> PipelineResult:
        """
        Run the full ReadAgent pipeline.

        Args:
            document: The full document text.
            question: The question/task to answer.
            options: Multiple choice options (optional).

        Returns:
            PipelineResult with answer, metrics, and intermediate data.
        """
        result = PipelineResult()
        self.llm.reset_stats()
        total_start = time.time()

        # Determine active mode
        mode_parts = []
        if self.config.fractal.enabled:
            mode_parts.append("fractal")
        if self.config.predictive.enabled:
            mode_parts.append("predictive")
        if self.config.differentiable.enabled:
            mode_parts.append("differentiable")
        result.mode = "+".join(mode_parts) if mode_parts else "base"

        logger.info(f"=== Running ReadAgent pipeline (mode: {result.mode}) ===")
        logger.info(f"Document: {count_words(document)} words")
        logger.info(f"Question: {question[:100]}...")

        # ─── Step 1: Episode Pagination ──────────────────────────────────
        t0 = time.time()
        pages = paginate_document(document, self.llm, self.config.pagination)
        result.pages = pages
        result.timings["pagination"] = time.time() - t0
        logger.info(f"Step 1 (Pagination): {len(pages)} pages in {result.timings['pagination']:.1f}s")

        # ─── Step 2: Gisting ────────────────────────────────────────────
        t0 = time.time()
        if self.config.predictive.enabled:
            # IMPROVEMENT 2: Predictive Task-Driven Gisting
            logger.info("Using PREDICTIVE TASK-DRIVEN gisting")
            gist_memory, importance_meta = predictive_gist_pages(
                pages, document, self.llm, self.config.predictive
            )
            result.metadata["importance"] = importance_meta
        else:
            # Standard unconditional gisting
            gist_memory = gist_pages(pages, self.llm, self.config.gisting)

        result.gist_memory = gist_memory
        result.compression_rate = gist_memory.compression_rate
        result.timings["gisting"] = time.time() - t0
        logger.info(
            f"Step 2 (Gisting): CR={gist_memory.compression_rate:.1f}% "
            f"in {result.timings['gisting']:.1f}s"
        )

        # ─── Step 2b: Fractal Tree (if needed) ──────────────────────────
        fractal_tree = None
        if self.config.fractal.enabled:
            gist_total_words = gist_memory.gist_word_count
            if gist_total_words > self.config.fractal.context_limit_words:
                t0 = time.time()
                logger.info(
                    f"FRACTAL: Gist memory ({gist_total_words} words) exceeds "
                    f"context limit ({self.config.fractal.context_limit_words}). "
                    f"Building fractal tree..."
                )
                fractal_tree = build_fractal_tree(
                    gist_memory, self.llm, self.config.fractal
                )
                result.timings["fractal_build"] = time.time() - t0
                result.metadata["fractal_levels"] = fractal_tree.max_level
                logger.info(
                    f"Step 2b (Fractal): {fractal_tree.max_level} levels "
                    f"in {result.timings['fractal_build']:.1f}s"
                )
            else:
                logger.info(
                    f"FRACTAL: Gist memory ({gist_total_words} words) fits in "
                    f"context ({self.config.fractal.context_limit_words}). "
                    f"No recursion needed."
                )

        # ─── Step 3: Lookup / Retrieval ──────────────────────────────────
        t0 = time.time()

        if self.config.differentiable.enabled:
            # IMPROVEMENT 3: Differentiable Gist Retrieval
            logger.info("Using DIFFERENTIABLE gist retrieval")
            retriever = self._get_diff_retriever()
            retriever.encode_gists(gist_memory.gists, gist_memory.page_indices)
            selected_pages = retriever.retrieve_pages(
                question, top_k=self.config.differentiable.top_k
            )
            expanded_memory = gist_memory.get_expanded_memory(pages, selected_pages)
            result.metadata["retrieval_method"] = "differentiable"

        elif fractal_tree is not None and fractal_tree.max_level > 0:
            # IMPROVEMENT 1: Fractal tree navigation
            logger.info("Using FRACTAL tree lookup")
            selected_pages, expanded_memory = fractal_lookup(
                fractal_tree, pages, gist_memory, question,
                self.llm, max_pages=self.config.lookup.max_lookups,
            )
            result.metadata["retrieval_method"] = "fractal"

        elif self.config.lookup.strategy == "sequential":
            # ReadAgent-S
            selected_pages, expanded_memory = sequential_lookup(
                gist_memory, pages, question, self.llm, self.config.lookup
            )
            result.metadata["retrieval_method"] = "sequential"

        else:
            # ReadAgent-P (default)
            selected_pages, expanded_memory = parallel_lookup(
                gist_memory, pages, question, self.llm, self.config.lookup
            )
            result.metadata["retrieval_method"] = "parallel"

        result.selected_pages = selected_pages
        result.expanded_memory = expanded_memory
        result.timings["lookup"] = time.time() - t0
        logger.info(
            f"Step 3 (Lookup): pages={selected_pages} "
            f"in {result.timings['lookup']:.1f}s"
        )

        # ─── Step 4: Response Generation ─────────────────────────────────
        t0 = time.time()
        answer = generate_response(expanded_memory, question, self.llm, options)
        result.answer = answer
        result.timings["response"] = time.time() - t0
        logger.info(f"Step 4 (Response): generated in {result.timings['response']:.1f}s")

        # ─── Final stats ─────────────────────────────────────────────────
        result.timings["total"] = time.time() - total_start
        result.token_usage = self.llm.get_usage_stats()

        logger.info(f"=== Pipeline complete ({result.mode}) ===")
        logger.info(f"Total time: {result.timings['total']:.1f}s")
        logger.info(f"Token usage: {result.token_usage}")
        logger.info(f"Answer: {answer[:200]}...")

        return result
