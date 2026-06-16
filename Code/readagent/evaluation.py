"""
Evaluation metrics for ReadAgent.
Implements ROUGE scores and LLM Rater (Strict + Permissive) from the base paper.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Lazy import rouge_score
_rouge_scorer = None


def _get_rouge_scorer():
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer as rs
        _rouge_scorer = rs
    return _rouge_scorer


STRICT_RATER_PROMPT = """After reading some text, John was given the following question about the text:
{question}

John's answer to the question was:
{model_response}

The ground truth answer was:
{reference}

Does John's answer agree with the ground truth answer?
Please answer YES or NO."""

PERMISSIVE_RATER_PROMPT = """After reading some text, John was given the following question about the text:
{question}

John's answer to the question was:
{model_response}

The ground truth answer was:
{reference}

Does John's answer agree with the ground truth answer?
Please answer "Yes", "Yes, partially", or "No". If John's response has any overlap with the ground truth answer, answer "Yes, partially". If John's response contains the ground truth answer, answer "Yes". If John's response is more specific than the ground truth answer, answer "Yes"."""


def compute_rouge(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F-measures."""
    rs = _get_rouge_scorer()
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def llm_rate(question, model_response, reference, llm) -> Dict[str, str]:
    """
    Use LLM as rater (Strict + Permissive) per the base paper.
    Returns dict with 'strict', 'permissive', and 'rating' keys.
    """
    strict_prompt = STRICT_RATER_PROMPT.format(
        question=question, model_response=model_response, reference=reference,
    )
    strict_resp = llm.generate(strict_prompt, max_tokens=64).strip().upper()

    permissive_prompt = PERMISSIVE_RATER_PROMPT.format(
        question=question, model_response=model_response, reference=reference,
    )
    permissive_resp = llm.generate(permissive_prompt, max_tokens=64).strip().upper()

    strict_match = "YES" in strict_resp and "NO" not in strict_resp
    permissive_partial = "PARTIALLY" in permissive_resp
    permissive_match = "YES" in permissive_resp and "PARTIALLY" not in permissive_resp

    if strict_match or permissive_match:
        rating = "exact_match"
    elif permissive_partial:
        rating = "partial_match"
    else:
        rating = "no_match"

    return {
        "strict": strict_resp,
        "permissive": permissive_resp,
        "rating": rating,
    }
