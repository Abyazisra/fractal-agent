"""
FractalAgent: Recursive Gisting for Higher Context Scaling.

IMPROVEMENT 1: When the concatenated gist memory exceeds the model's
context window, FractalAgent recursively creates "gists of gists"
(meta-gists) forming a fractal tree. During lookup, the agent navigates
from the top level down to identify relevant branches.
"""

import logging
from typing import List, Tuple

from .llm import LLMClient
from .config import FractalConfig
from .utils import (
    Page, GistMemory, FractalNode, FractalTree,
    count_words, parse_fractal_branches,
)

logger = logging.getLogger(__name__)

FRACTAL_GISTING_PROMPT = """Below are several gist summaries from consecutive sections of a long document. Please combine and shorten them into a single, concise meta-summary that captures the key points from all sections.

Just give me the shortened combined version. DO NOT explain your reason.

Sections:
{gist_group}"""

FRACTAL_LOOKUP_PROMPT = """The following is a high-level summary of different branches of a very long document, followed by a question about the document.

Please identify which branch(es) contain information most relevant to answering the question.
Respond with the branch number(s) in brackets, e.g., "I want to explore Branch [2, 5] because ..."

DO NOT select more branches than necessary.
You don't need to answer the question yet.

Document Overview:
{top_level_text}

Question:
{question}"""

FRACTAL_DRILL_PROMPT = """The following are more detailed summaries from one branch of a long document, followed by a question.

Please identify which section(s) are most relevant to the question.
Respond with the section number(s) in brackets, e.g., "I want to look at Node [3, 7] because ..."

Sections in this branch:
{branch_text}

Question:
{question}"""


def build_fractal_tree(
    gist_memory: GistMemory,
    llm: LLMClient,
    config: FractalConfig,
) -> FractalTree:
    """
    Build a fractal tree by recursively gisting groups of gists.

    When the Level-0 gist memory exceeds the context limit, we group
    adjacent gists and create Level-1 meta-gists. This continues until
    the top level fits within the context window.

    Args:
        gist_memory: The original Level-0 gist memory.
        llm: The LLM client.
        config: FractalAgent configuration.

    Returns:
        FractalTree with multiple levels of gists.
    """
    tree = FractalTree()

    # Level 0: original gists
    level_0_nodes = []
    for idx, gist in zip(gist_memory.page_indices, gist_memory.gists):
        node = FractalNode(level=0, index=idx, gist=gist)
        level_0_nodes.append(node)
    tree.levels[0] = level_0_nodes

    current_level = 0
    current_nodes = level_0_nodes

    while True:
        # Check if current level fits in context
        total_words = sum(n.word_count for n in current_nodes)
        logger.info(
            f"Fractal Level {current_level}: {len(current_nodes)} nodes, "
            f"{total_words} words (limit: {config.context_limit_words})"
        )

        if total_words <= config.context_limit_words:
            tree.max_level = current_level
            logger.info(f"Fractal tree complete at level {current_level}")
            break

        if current_level >= config.max_levels:
            logger.warning(f"Reached max fractal depth ({config.max_levels})")
            tree.max_level = current_level
            break

        # Create next level by grouping and meta-gisting
        next_level = current_level + 1
        next_nodes = []
        meta_index = 0

        for i in range(0, len(current_nodes), config.group_size):
            group = current_nodes[i:i + config.group_size]
            children_indices = [n.index for n in group]

            # Combine gists for this group
            gist_group_text = "\n\n".join(
                f"Section {n.index}: {n.gist}" for n in group
            )

            # Generate meta-gist
            prompt = FRACTAL_GISTING_PROMPT.format(gist_group=gist_group_text)
            meta_gist = llm.generate(prompt, max_tokens=1024)
            meta_gist = meta_gist.strip()

            meta_node = FractalNode(
                level=next_level,
                index=meta_index,
                gist=meta_gist,
                children_indices=children_indices,
            )
            next_nodes.append(meta_node)

            logger.info(
                f"Meta-gist L{next_level}-{meta_index}: "
                f"combined {len(group)} nodes -> {meta_node.word_count} words"
            )
            meta_index += 1

        tree.levels[next_level] = next_nodes
        current_level = next_level
        current_nodes = next_nodes

    return tree


def fractal_lookup(
    tree: FractalTree,
    pages: List[Page],
    gist_memory: GistMemory,
    question: str,
    llm: LLMClient,
    max_pages: int = 3,
) -> Tuple[List[int], str]:
    """
    Navigate the fractal tree to find relevant pages.

    Starting from the top level, the agent selects relevant branches,
    then drills down level by level until reaching the original page
    gists (Level 0), then expands those pages.

    Args:
        tree: The fractal tree.
        pages: All original pages.
        gist_memory: The Level-0 gist memory.
        question: The task/question.
        llm: The LLM client.
        max_pages: Maximum pages to expand.

    Returns:
        Tuple of (selected page indices, expanded memory text).
    """
    if tree.max_level == 0:
        # No recursion was needed, fall back to standard lookup
        logger.info("Fractal tree has only 1 level, using standard lookup")
        from .lookup import parallel_lookup
        from .config import LookupConfig
        return parallel_lookup(
            gist_memory, pages, question, llm,
            LookupConfig(max_lookups=max_pages)
        )

    # Start from top level
    top_text = tree.get_top_level_text()

    prompt = FRACTAL_LOOKUP_PROMPT.format(
        top_level_text=top_text,
        question=question,
    )
    response = llm.generate(prompt, max_tokens=512)
    selected_branches = parse_fractal_branches(response)

    logger.info(f"Fractal top-level: selected branches {selected_branches}")

    # Drill down through levels
    candidate_page_indices = set()

    for branch_idx in selected_branches:
        current_level = tree.max_level
        current_indices = [branch_idx]

        while current_level > 0:
            # Get children of selected nodes at this level
            next_indices = []
            for idx in current_indices:
                for node in tree.levels.get(current_level, []):
                    if node.index == idx:
                        next_indices.extend(node.children_indices)
                        break

            if current_level - 1 == 0:
                # We're at the page gist level
                candidate_page_indices.update(next_indices)
            else:
                # Drill down further — ask LLM which sub-branches
                branch_text = ""
                for child_idx in next_indices:
                    for node in tree.levels.get(current_level - 1, []):
                        if node.index == child_idx:
                            branch_text += f"Node {child_idx}: {node.gist}\n\n"
                            break

                if branch_text:
                    drill_prompt = FRACTAL_DRILL_PROMPT.format(
                        branch_text=branch_text,
                        question=question,
                    )
                    drill_response = llm.generate(drill_prompt, max_tokens=512)
                    drilled = parse_fractal_branches(drill_response)
                    if drilled:
                        next_indices = drilled

                current_indices = next_indices

            current_level -= 1

    # Select top pages from candidates
    selected_pages = list(candidate_page_indices)[:max_pages]

    logger.info(f"Fractal lookup final pages: {selected_pages}")

    # Build expanded memory using the full gist memory with expansions
    expanded_memory = gist_memory.get_expanded_memory(pages, selected_pages)

    return selected_pages, expanded_memory
