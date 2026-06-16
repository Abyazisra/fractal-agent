"""
Utility functions for text processing, word counting, and compression metrics.
"""

import re
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class Page:
    """Represents an episode/page in the document."""
    index: int
    text: str
    word_count: int = 0
    start_paragraph: int = 0
    end_paragraph: int = 0

    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = count_words(self.text)


@dataclass
class GistMemory:
    """Represents the gist memory — an ordered collection of gists."""
    gists: List[str] = field(default_factory=list)
    page_indices: List[int] = field(default_factory=list)
    compression_rate: float = 0.0
    original_word_count: int = 0
    gist_word_count: int = 0

    def get_full_gist_text(self) -> str:
        """Concatenate all gists with page tags into a single string."""
        parts = []
        for idx, gist in zip(self.page_indices, self.gists):
            parts.append(f"<Page {idx}>\n{gist}")
        return "\n\n".join(parts)

    def get_gist_for_page(self, page_idx: int) -> str:
        """Get the gist for a specific page index."""
        for idx, gist in zip(self.page_indices, self.gists):
            if idx == page_idx:
                return gist
        return ""

    def replace_gist_with_page(self, page_idx: int, page_text: str) -> str:
        """
        Create an expanded memory where the gist at page_idx
        is replaced with the full page text. Returns the full text.
        """
        parts = []
        for idx, gist in zip(self.page_indices, self.gists):
            if idx == page_idx:
                parts.append(f"<Page {idx}> [EXPANDED]\n{page_text}")
            else:
                parts.append(f"<Page {idx}>\n{gist}")
        return "\n\n".join(parts)

    def get_expanded_memory(self, pages: List[Page], expand_indices: List[int]) -> str:
        """
        Create memory with selected pages expanded (raw text replaces gist).

        Args:
            pages: All original pages.
            expand_indices: Which page indices to expand.

        Returns:
            The full memory text with selected pages expanded.
        """
        page_map = {p.index: p.text for p in pages}
        parts = []
        for idx, gist in zip(self.page_indices, self.gists):
            if idx in expand_indices and idx in page_map:
                parts.append(f"<Page {idx}> [EXPANDED]\n{page_map[idx]}")
            else:
                parts.append(f"<Page {idx}>\n{gist}")
        return "\n\n".join(parts)


@dataclass
class FractalNode:
    """A node in the fractal gist tree."""
    level: int
    index: int
    gist: str
    children_indices: List[int] = field(default_factory=list)
    word_count: int = 0

    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = count_words(self.gist)


@dataclass
class FractalTree:
    """The multi-level fractal tree of gists."""
    levels: dict = field(default_factory=dict)  # level_num -> list of FractalNodes
    max_level: int = 0

    def get_top_level_text(self) -> str:
        """Get the concatenated text of the top-level gists."""
        if self.max_level not in self.levels:
            return ""
        parts = []
        for node in self.levels[self.max_level]:
            parts.append(f"<Branch {node.index}>\n{node.gist}")
        return "\n\n".join(parts)

    def get_children_text(self, level: int, branch_idx: int) -> str:
        """Get concatenated text of children of a specific node."""
        if level not in self.levels:
            return ""
        node = None
        for n in self.levels[level]:
            if n.index == branch_idx:
                node = n
                break
        if node is None or level - 1 not in self.levels:
            return ""

        parts = []
        for child_idx in node.children_indices:
            for child_node in self.levels[level - 1]:
                if child_node.index == child_idx:
                    parts.append(f"<Node {child_idx}>\n{child_node.gist}")
                    break
        return "\n\n".join(parts)


def count_words(text: str) -> int:
    """Count the number of words in text."""
    return len(text.split())


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs (by double newlines or single newlines with blank lines)."""
    # Split on double newlines or more
    paragraphs = re.split(r'\n\s*\n', text.strip())
    # Filter out empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def compute_compression_rate(original_words: int, compressed_words: int) -> float:
    """
    Compute compression rate as defined in the paper:
    CR = 100 * (1 - compressed_words / original_words)
    """
    if original_words == 0:
        return 0.0
    return 100.0 * (1.0 - compressed_words / original_words)


def insert_pause_tags(paragraphs: List[str], min_words: int) -> Tuple[str, List[int]]:
    """
    Insert numbered pause tags between paragraphs after min_words threshold.

    Args:
        paragraphs: List of paragraph texts.
        min_words: Minimum words before inserting first tag.

    Returns:
        Tuple of (text_with_tags, list_of_valid_tag_numbers)
    """
    accumulated_words = 0
    result_parts = []
    valid_tags = []
    tag_number = 0

    for i, para in enumerate(paragraphs):
        result_parts.append(para)
        accumulated_words += count_words(para)

        if i < len(paragraphs) - 1:  # Don't add tag after last paragraph
            if accumulated_words >= min_words:
                tag_number += 1
                result_parts.append(f"\n<{tag_number}>\n")
                valid_tags.append(tag_number)
            else:
                result_parts.append("\n\n")

    return "\n".join(result_parts), valid_tags


def parse_break_point(response: str) -> int:
    """
    Parse the LLM's chosen break point from its response.
    Looks for patterns like 'Break point: <5>' or 'Break point: ⟨5⟩' or just a number.
    """
    # Try to find break point pattern
    patterns = [
        r'Break point:\s*[<⟨](\d+)[>⟩]',
        r'break point:\s*[<⟨](\d+)[>⟩]',
        r'[<⟨](\d+)[>⟩]',
        r'Break point:\s*(\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return int(match.group(1))

    # Fallback: find any number in the response
    numbers = re.findall(r'\d+', response)
    if numbers:
        return int(numbers[0])

    return -1


def parse_lookup_pages(response: str) -> List[int]:
    """
    Parse page numbers from the LLM's lookup response.
    Handles formats like 'Page [2, 3, 7]' or 'Page 5' or 'Page [8]'.
    """
    # Check for STOP
    if "STOP" in response.upper():
        return []

    # Try to find bracketed list
    bracket_match = re.search(r'\[([^\]]+)\]', response)
    if bracket_match:
        numbers = re.findall(r'\d+', bracket_match.group(1))
        return [int(n) for n in numbers]

    # Try 'Page X' pattern
    page_matches = re.findall(r'Page\s+(\d+)', response, re.IGNORECASE)
    if page_matches:
        return [int(n) for n in page_matches]

    # Fallback
    numbers = re.findall(r'\d+', response)
    return [int(n) for n in numbers[:5]]  # Cap at 5


def parse_fractal_branches(response: str) -> List[int]:
    """Parse branch indices from fractal lookup response."""
    bracket_match = re.search(r'\[([^\]]+)\]', response)
    if bracket_match:
        numbers = re.findall(r'\d+', bracket_match.group(1))
        return [int(n) for n in numbers]

    branch_matches = re.findall(r'Branch\s+(\d+)', response, re.IGNORECASE)
    if branch_matches:
        return [int(n) for n in branch_matches]

    return []
