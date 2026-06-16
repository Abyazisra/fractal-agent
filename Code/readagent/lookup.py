"""
Interactive Lookup Module.

Implements the third step of ReadAgent: using gist memory and the task
to decide which original pages to look up for detailed reading.
Supports both parallel (ReadAgent-P) and sequential (ReadAgent-S) strategies.
"""

import logging
from typing import List, Tuple

from .llm import LLMClient
from .config import LookupConfig
from .utils import Page, GistMemory, parse_lookup_pages

logger = logging.getLogger(__name__)

# ─── Parallel Lookup Prompt ─────────────────────────────────────────────────
PARALLEL_LOOKUP_PROMPT = """The following text is what you remember from reading a document and a question related to it.

You may read 1 to {max_pages} page(s) of the document again to refresh your memory to prepare yourself for the question.

Please respond with which page(s) you would like to read.
For example, if you only need to read Page 8, respond with "I want to look up Page [8] to ..."; if you would like to read Page 7 and 12, respond with "I want to look up Page [7, 12] to ...".

DO NOT select more pages if you don't need to.
You don't need to answer the question yet.

Text:
{gist_memory}

Question:
{question}"""

# ─── Sequential Lookup Prompt ───────────────────────────────────────────────
SEQUENTIAL_LOOKUP_PROMPT = """The following text is what you remember from reading a document, followed by a question about the document.

You may read multiple pages of the document again to refresh your memory and prepare to answer the question.
Each page that you re-read can significantly improve your chance of answering the question correctly.

Please specify a SINGLE page you would like to read again or say "STOP".
To read a page again, respond with "Page $PAGE_NUM", replacing $PAGE_NUM with the target page number.
You can only specify a SINGLE page in your response at this time.
To stop, simply say "STOP". DO NOT answer the question in your response.

Text:
{expanded_memory}

Pages re-read already (DO NOT ask to read them again):
{already_read}

Question:
{question}

Specify a SINGLE page to read again, or say STOP:"""

# ─── Response Prompts ───────────────────────────────────────────────────────
RESPONSE_FREEFORM_PROMPT = """{expanded_memory}

Question: {question}
Answer the question based on the above passage and retrieved pages. Your answer should be short and concise."""

RESPONSE_MCQ_PROMPT = """Read the following article and answer a multiple choice question. For example, if (C) is correct, answer with "Answer: (C) ..."

Article: {expanded_memory}

Question: {question} {options}"""


def parallel_lookup(
    gist_memory: GistMemory,
    pages: List[Page],
    question: str,
    llm: LLMClient,
    config: LookupConfig,
) -> Tuple[List[int], str]:
    """
    ReadAgent-P: Parallel lookup strategy.

    The LLM sees all gists and selects multiple pages at once to expand.

    Args:
        gist_memory: The gist memory.
        pages: All original pages.
        question: The task/question to answer.
        llm: The LLM client.
        config: Lookup configuration.

    Returns:
        Tuple of (selected page indices, expanded memory text).
    """
    gist_text = gist_memory.get_full_gist_text()

    prompt = PARALLEL_LOOKUP_PROMPT.format(
        max_pages=config.max_lookups,
        gist_memory=gist_text,
        question=question,
    )

    response = llm.generate(prompt, max_tokens=512)
    selected_pages = parse_lookup_pages(response)

    # Validate page indices
    valid_indices = set(p.index for p in pages)
    selected_pages = [p for p in selected_pages if p in valid_indices]
    selected_pages = selected_pages[:config.max_lookups]

    logger.info(f"Parallel lookup selected pages: {selected_pages}")

    # Create expanded memory
    expanded_memory = gist_memory.get_expanded_memory(pages, selected_pages)

    return selected_pages, expanded_memory


def sequential_lookup(
    gist_memory: GistMemory,
    pages: List[Page],
    question: str,
    llm: LLMClient,
    config: LookupConfig,
) -> Tuple[List[int], str]:
    """
    ReadAgent-S: Sequential lookup strategy.

    The LLM selects one page at a time, seeing previously expanded pages
    before deciding the next one.

    Args:
        gist_memory: The gist memory.
        pages: All original pages.
        question: The task/question to answer.
        llm: The LLM client.
        config: Lookup configuration.

    Returns:
        Tuple of (selected page indices, expanded memory text).
    """
    selected_pages = []
    valid_indices = set(p.index for p in pages)
    page_map = {p.index: p.text for p in pages}

    for step in range(config.max_lookups):
        # Build current expanded memory
        expanded_memory = gist_memory.get_expanded_memory(pages, selected_pages)

        already_read = ", ".join(str(p) for p in selected_pages) if selected_pages else "None"

        prompt = SEQUENTIAL_LOOKUP_PROMPT.format(
            expanded_memory=expanded_memory,
            already_read=already_read,
            question=question,
        )

        response = llm.generate(prompt, max_tokens=256)

        # Check for STOP
        if "STOP" in response.upper():
            logger.info(f"Sequential lookup stopped at step {step + 1}")
            break

        # Parse the single page
        page_nums = parse_lookup_pages(response)

        if not page_nums:
            logger.info(f"Sequential lookup: no page found in response, stopping")
            break

        page_num = page_nums[0]

        if page_num not in valid_indices or page_num in selected_pages:
            logger.warning(f"Invalid or already-read page {page_num}, stopping")
            break

        selected_pages.append(page_num)
        logger.info(f"Sequential lookup step {step + 1}: selected Page {page_num}")

    # Build final expanded memory
    expanded_memory = gist_memory.get_expanded_memory(pages, selected_pages)

    logger.info(f"Sequential lookup selected pages: {selected_pages}")
    return selected_pages, expanded_memory


def generate_response(
    expanded_memory: str,
    question: str,
    llm: LLMClient,
    options: str = None,
) -> str:
    """
    Generate the final response using expanded memory and the question.

    Args:
        expanded_memory: The gist memory with selected pages expanded.
        question: The task/question.
        llm: The LLM client.
        options: Multiple choice options (if applicable).

    Returns:
        The LLM's answer.
    """
    if options:
        prompt = RESPONSE_MCQ_PROMPT.format(
            expanded_memory=expanded_memory,
            question=question,
            options=options,
        )
    else:
        prompt = RESPONSE_FREEFORM_PROMPT.format(
            expanded_memory=expanded_memory,
            question=question,
        )

    response = llm.generate(prompt, max_tokens=1024)
    return response.strip()
