"""
Configuration settings for ReadAgent and its extensions.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PaginationConfig:
    """Hyperparameters for episode pagination."""
    min_words: int = 280
    max_words: int = 600


@dataclass
class GistingConfig:
    """Hyperparameters for memory gisting."""
    use_shorten_prompt: bool = True  # Use "shorten" vs "summarize"


@dataclass
class LookupConfig:
    """Hyperparameters for interactive lookup."""
    strategy: str = "parallel"  # "parallel" or "sequential"
    max_lookups: int = 5


@dataclass
class FractalConfig:
    """Hyperparameters for FractalAgent recursive gisting."""
    enabled: bool = False
    group_size: int = 5  # Number of gists to combine at each level
    max_levels: int = 5  # Maximum recursion depth
    context_limit_words: int = 6000  # Trigger fractal if gists exceed this


@dataclass
class PredictiveGistingConfig:
    """Hyperparameters for predictive task-driven gisting."""
    enabled: bool = False
    num_hypothesized_tasks: int = 5  # Number of tasks to predict
    detail_levels: dict = field(default_factory=lambda: {
        "HIGH": 0.9,    # Preserve ~90% of content
        "MEDIUM": 0.5,  # Standard ~50% compression
        "LOW": 0.2,     # Aggressive ~20% retention
    })


@dataclass
class DifferentiableRetrievalConfig:
    """Hyperparameters for differentiable gist retrieval."""
    enabled: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence-transformer model
    adapter_hidden_dim: int = 256
    top_k: int = 2  # Max pages to retrieve (adaptive selection may return fewer)
    use_cross_attention: bool = False  # Cosine similarity works better without training


@dataclass
class LLMConfig:
    """LLM API configuration."""
    api_key: Optional[str] = None
    model: str = "mistral-small-latest"
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("MISTRAL_API_KEY", "")


@dataclass
class ReadAgentConfig:
    """Master configuration for the full ReadAgent pipeline."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    pagination: PaginationConfig = field(default_factory=PaginationConfig)
    gisting: GistingConfig = field(default_factory=GistingConfig)
    lookup: LookupConfig = field(default_factory=LookupConfig)
    fractal: FractalConfig = field(default_factory=FractalConfig)
    predictive: PredictiveGistingConfig = field(default_factory=PredictiveGistingConfig)
    differentiable: DifferentiableRetrievalConfig = field(default_factory=DifferentiableRetrievalConfig)

    @classmethod
    def base(cls, api_key: str = None) -> "ReadAgentConfig":
        """Standard ReadAgent configuration (no improvements)."""
        return cls(llm=LLMConfig(api_key=api_key))

    @classmethod
    def with_fractal(cls, api_key: str = None) -> "ReadAgentConfig":
        """ReadAgent + FractalAgent."""
        return cls(
            llm=LLMConfig(api_key=api_key),
            fractal=FractalConfig(enabled=True),
        )

    @classmethod
    def with_predictive(cls, api_key: str = None) -> "ReadAgentConfig":
        """ReadAgent + Predictive Task-Driven Gisting."""
        return cls(
            llm=LLMConfig(api_key=api_key),
            predictive=PredictiveGistingConfig(enabled=True),
        )

    @classmethod
    def with_differentiable(cls, api_key: str = None) -> "ReadAgentConfig":
        """ReadAgent + Differentiable Gist Retrieval."""
        return cls(
            llm=LLMConfig(api_key=api_key),
            differentiable=DifferentiableRetrievalConfig(enabled=True),
        )

    @classmethod
    def full(cls, api_key: str = None) -> "ReadAgentConfig":
        """ReadAgent + all three improvements."""
        return cls(
            llm=LLMConfig(api_key=api_key),
            fractal=FractalConfig(enabled=True),
            predictive=PredictiveGistingConfig(enabled=True),
            differentiable=DifferentiableRetrievalConfig(enabled=True),
        )
