"""
Memory Gisting Module.

Implements the second step of ReadAgent: compressing each page into
a short episodic gist memory using the LLM.
"""

import logging
from typing import List

from .llm import LLMClient
from .config import GistingConfig
from .utils import Page, GistMemory, count_words, compute_compression_rate

logger = logging.getLogger(__name__)

GISTING_PROMPT = """Please shorten the following passage.
Just give me a shortened version. DO NOT explain your reason.

Passage:
{page_text}"""

SUMMARIZE_PROMPT = """Please summarize the following passage concisely.
Just give me a summary. DO NOT explain your reason.

Passage:
{page_text}"""


def gist_pages(
    pages: List[Page],
    llm: LLMClient,
    config: GistingConfig,
) -> GistMemory:
    """
    Compress each page into a short gist using the LLM.

    Uses "shorten" rather than "summarize" to preserve narrative flow,
    as recommended in the base paper.

    Args:
        pages: List of Page objects from pagination.
        llm: The LLM client for inference.
        config: Gisting configuration.

    Returns:
        GistMemory containing all gists with metadata.
    """
    gists = []
    page_indices = []
    total_original_words = 0
    total_gist_words = 0

    for page in pages:
        # Choose prompt based on config
        if config.use_shorten_prompt:
            prompt = GISTING_PROMPT.format(page_text=page.text)
        else:
            prompt = SUMMARIZE_PROMPT.format(page_text=page.text)

        gist = llm.generate(prompt, max_tokens=1024)
        gist = gist.strip()

        gist_wc = count_words(gist)
        total_original_words += page.word_count
        total_gist_words += gist_wc

        gists.append(gist)
        page_indices.append(page.index)

        logger.info(
            f"Page {page.index}: {page.word_count} words -> {gist_wc} words "
            f"({compute_compression_rate(page.word_count, gist_wc):.1f}% CR)"
        )

    overall_cr = compute_compression_rate(total_original_words, total_gist_words)

    gist_memory = GistMemory(
        gists=gists,
        page_indices=page_indices,
        compression_rate=overall_cr,
        original_word_count=total_original_words,
        gist_word_count=total_gist_words,
    )

    logger.info(
        f"Gisting complete: {total_original_words} -> {total_gist_words} words "
        f"(CR: {overall_cr:.1f}%)"
    )

    return gist_memory
