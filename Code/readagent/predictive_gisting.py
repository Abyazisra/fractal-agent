"""
Predictive Task-Driven Gisting Module.

IMPROVEMENT 2: Instead of unconditionally compressing text, this module
first predicts likely downstream tasks, then uses those tasks to guide
the gisting — telling the LLM exactly what information to preserve.
This produces task-aware gists that retain critical details while still
achieving meaningful compression.
"""

import logging
from typing import List, Dict, Tuple

from .llm import LLMClient
from .config import PredictiveGistingConfig
from .utils import Page, GistMemory, count_words, compute_compression_rate

logger = logging.getLogger(__name__)

TASK_FORECAST_PROMPT = """You are a task prediction agent. Given the beginning of a document, predict what kinds of questions a reader might be asked about this document.

Generate {num_tasks} specific, diverse questions covering different aspects (factual details, character motivations, key events, specific numbers/dates, cause-effect relationships).

Document beginning:
{document_start}

List exactly {num_tasks} predicted questions, one per line, starting each with "Q:"."""

TASK_AWARE_GIST_PROMPT = """Please shorten the following passage while making sure to preserve information relevant to these anticipated questions:

{task_list}

Preserve all specific names, numbers, dates, direct quotes, key facts, and cause-effect relationships that could help answer such questions. Remove only redundant phrasing, filler words, and stylistic repetition.

Just give me the shortened version. DO NOT explain your reason.

Passage:
{page_text}"""


def forecast_tasks(document_text, llm, config):
    """Predict likely downstream tasks from document content."""
    words = document_text.split()
    doc_start = " ".join(words[:500])
    prompt = TASK_FORECAST_PROMPT.format(
        num_tasks=config.num_hypothesized_tasks,
        document_start=doc_start,
    )
    response = llm.generate(prompt, max_tokens=1024)

    tasks = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("Q:"):
            tasks.append(line[2:].strip())
        elif line and line[0].isdigit() and "." in line[:4]:
            # Handle "1. question" format
            parts = line.split(".", 1)
            if len(parts) > 1:
                tasks.append(parts[1].strip())

    # Fallback: if parsing failed, use lines directly
    if not tasks:
        tasks = [line.strip() for line in response.strip().split("\n")
                 if line.strip() and len(line.strip()) > 10]

    tasks = tasks[:config.num_hypothesized_tasks]
    logger.info(f"Forecasted {len(tasks)} hypothesized tasks")
    for i, t in enumerate(tasks):
        logger.debug(f"  Task {i+1}: {t}")
    return tasks


def predictive_gist_pages(pages, document_text, llm, config):
    """
    Task-driven gisting: forecast likely questions, then gist each page
    with explicit instructions to preserve task-relevant information.

    The key difference from base gisting: the LLM knows WHAT to preserve,
    not just "shorten this." This produces gists that retain critical
    details (names, numbers, quotes, causal chains) that standard
    unconditional gisting would discard.

    Returns: (GistMemory, importance_metadata_list)
    """
    # Step 1: Forecast tasks
    tasks = forecast_tasks(document_text, llm, config)
    if not tasks:
        logger.warning("No tasks forecasted, falling back to standard gisting")
        from .gisting import gist_pages
        from .config import GistingConfig
        return gist_pages(pages, llm, GistingConfig()), []

    # Format task list for the prompt
    task_list = "\n".join(f"- {t}" for t in tasks)

    # Step 2: Gist each page with task-aware prompt
    gists, page_indices, metadata = [], [], []
    total_orig, total_gist = 0, 0

    for page in pages:
        prompt = TASK_AWARE_GIST_PROMPT.format(
            task_list=task_list,
            page_text=page.text,
        )
        gist = llm.generate(prompt, max_tokens=1024).strip()

        gist_wc = count_words(gist)
        total_orig += page.word_count
        total_gist += gist_wc
        gists.append(gist)
        page_indices.append(page.index)
        metadata.append({
            "page_index": page.index,
            "original_words": page.word_count,
            "gist_words": gist_wc,
            "compression_rate": compute_compression_rate(page.word_count, gist_wc),
        })
        logger.info(f"Page {page.index} [TASK-AWARE]: {page.word_count}->{gist_wc} words")

    cr = compute_compression_rate(total_orig, total_gist)
    gist_memory = GistMemory(
        gists=gists, page_indices=page_indices,
        compression_rate=cr, original_word_count=total_orig, gist_word_count=total_gist,
    )
    logger.info(f"Predictive gisting: {total_orig}->{total_gist} words (CR: {cr:.1f}%)")

    # Store metadata including the predicted tasks
    metadata_with_tasks = {
        "predicted_tasks": tasks,
        "pages": metadata,
    }
    return gist_memory, metadata_with_tasks
