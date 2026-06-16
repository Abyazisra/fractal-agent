"""
Episode Pagination Module.

Implements the first step of ReadAgent: breaking a long document into
variable-length episodes (pages) using LLM-guided natural pause points.
"""

import logging
from typing import List

from .llm import LLMClient
from .config import PaginationConfig
from .utils import (
    Page, count_words, split_into_paragraphs,
    insert_pause_tags, parse_break_point,
)

logger = logging.getLogger(__name__)

PAGINATION_PROMPT = """You are given a passage that is taken from a larger text (article, book, etc.) and some numbered labels between the paragraphs in the passage.

Numbered labels are in angle brackets. For example, if the label number is 19, it shows as <19> in text.

Please choose a label where it is natural to break reading. The label can be a scene transition, the end of a dialogue, the end of an argument, a narrative transition, etc.

Please answer with the break point label and explain.
For example, if <57> is a good point to break, answer with "Break point: <57>
Because ..."

Passage:
{passage}"""


def paginate_document(
    text: str,
    llm: LLMClient,
    config: PaginationConfig,
) -> List[Page]:
    """
    Split a document into episodes (pages) using LLM-guided pagination.

    The LLM reads chunks of text and decides where natural break points are
    (scene transitions, end of dialogue, narrative shifts, etc.).

    Args:
        text: The full document text.
        llm: The LLM client for inference.
        config: Pagination hyperparameters (min_words, max_words).

    Returns:
        List of Page objects representing the episodes.
    """
    paragraphs = split_into_paragraphs(text)

    if not paragraphs:
        return [Page(index=0, text=text)]

    pages = []
    current_start = 0  # Index into paragraphs
    page_index = 0

    while current_start < len(paragraphs):
        # Collect paragraphs up to max_words
        current_end = current_start
        accumulated_words = 0
        chunk_paragraphs = []

        while current_end < len(paragraphs) and accumulated_words < config.max_words:
            para = paragraphs[current_end]
            accumulated_words += count_words(para)
            chunk_paragraphs.append(para)
            current_end += 1

        # If we've consumed all remaining paragraphs, this is the last page
        if current_end >= len(paragraphs):
            page_text = "\n\n".join(chunk_paragraphs)
            pages.append(Page(
                index=page_index,
                text=page_text,
                start_paragraph=current_start,
                end_paragraph=current_end - 1,
            ))
            break

        # Insert pause tags and ask LLM where to break
        tagged_text, valid_tags = insert_pause_tags(
            chunk_paragraphs, config.min_words
        )

        if not valid_tags:
            # No valid tags (chunk is too short for tags), take the whole chunk
            page_text = "\n\n".join(chunk_paragraphs)
            pages.append(Page(
                index=page_index,
                text=page_text,
                start_paragraph=current_start,
                end_paragraph=current_end - 1,
            ))
            current_start = current_end
            page_index += 1
            continue

        # Ask LLM for break point
        prompt = PAGINATION_PROMPT.format(passage=tagged_text)
        response = llm.generate(prompt, max_tokens=256)
        break_point = parse_break_point(response)

        logger.info(f"Page {page_index}: LLM chose break point <{break_point}>")

        # Validate break point
        if break_point not in valid_tags:
            # Use the last valid tag as fallback
            break_point = valid_tags[-1]
            logger.warning(f"Invalid break point, using fallback: <{break_point}>")

        # Determine which paragraphs go into this page
        # Tags are numbered starting from 1, and placed after paragraphs
        # that have accumulated enough words (>= min_words).
        # We need to map tag numbers back to paragraph indices.
        tag_to_para_idx = {}
        word_acc = 0
        tag_counter = 0
        for i, para in enumerate(chunk_paragraphs):
            word_acc += count_words(para)
            if word_acc >= config.min_words and i < len(chunk_paragraphs) - 1:
                tag_counter += 1
                tag_to_para_idx[tag_counter] = i

        if break_point in tag_to_para_idx:
            end_para_local = tag_to_para_idx[break_point]
            page_paras = chunk_paragraphs[:end_para_local + 1]
        else:
            # Fallback: use all paragraphs in chunk
            page_paras = chunk_paragraphs
            end_para_local = len(chunk_paragraphs) - 1

        page_text = "\n\n".join(page_paras)
        pages.append(Page(
            index=page_index,
            text=page_text,
            start_paragraph=current_start,
            end_paragraph=current_start + end_para_local,
        ))

        current_start = current_start + end_para_local + 1
        page_index += 1

    logger.info(f"Pagination complete: {len(pages)} pages from {len(paragraphs)} paragraphs")
    return pages


def paginate_uniform(text: str, config: PaginationConfig) -> List[Page]:
    """
    Simple uniform-length pagination (baseline, no LLM needed).
    Splits text into chunks of approximately max_words each.

    Args:
        text: The full document text.
        config: Pagination hyperparameters.

    Returns:
        List of Page objects.
    """
    paragraphs = split_into_paragraphs(text)
    pages = []
    current_paras = []
    current_words = 0
    page_index = 0
    para_start = 0

    for i, para in enumerate(paragraphs):
        para_words = count_words(para)

        if current_words + para_words > config.max_words and current_paras:
            page_text = "\n\n".join(current_paras)
            pages.append(Page(
                index=page_index,
                text=page_text,
                start_paragraph=para_start,
                end_paragraph=i - 1,
            ))
            page_index += 1
            current_paras = [para]
            current_words = para_words
            para_start = i
        else:
            current_paras.append(para)
            current_words += para_words

    # Last page
    if current_paras:
        page_text = "\n\n".join(current_paras)
        pages.append(Page(
            index=page_index,
            text=page_text,
            start_paragraph=para_start,
            end_paragraph=len(paragraphs) - 1,
        ))

    return pages
