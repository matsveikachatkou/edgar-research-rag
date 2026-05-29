"""
eval/evaluate.py — RAG evaluation module.

Evaluates three metrics per question/answer/context triple:
    1. Context precision   — are the retrieved chunks relevant to the question?
    2. Answer faithfulness — does the answer stay within the retrieved context?
    3. Answer relevance    — does the answer actually address the question?

Each metric is scored 0-1 by an LLM judge using structured output.
This is a lightweight RAGAS-inspired implementation with no external dependencies.

Usage:
    uv run python eval/evaluate.py --ticker AAPL
    uv run python eval/evaluate.py --ticker AAPL --output eval_results.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion
from tenacity import retry, wait_exponential

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from answer import answer_question, fetch_context
from models.research import EvalResult, EvalSample, Result

MODEL = "openai/gpt-4.1-mini"
WAIT = wait_exponential(multiplier=1, min=10, max=240)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [eval] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# Default evaluation questions per domain

DEFAULT_QUESTIONS = [
    "What was the company's revenue in the most recent quarter?",
    "What are the main risk factors mentioned in the filing?",
    "What is management's outlook for the next quarter?",
    "How did operating margins change year over year?",
    "What are the key growth drivers mentioned by management?",
    "What is the company's current cash position?",
    "Were there any significant one-time items in the period?",
    "What did management say about competition?",
]


# Metric 1: Context precision


@retry(wait=WAIT)
def score_context_precision(question: str, chunks: list[Result]) -> float:
    """
    Score how relevant the retrieved chunks are to the question.
    Returns 0-1: fraction of chunks judged relevant by the LLM.
    """
    if not chunks:
        return 0.0

    chunk_list = "\n\n".join(
        f"CHUNK {i+1}:\n{c.page_content[:400]}"
        for i, c in enumerate(chunks)
    )

    prompt = f"""You are evaluating a RAG retrieval system for financial documents.

Question: {question}

Retrieved chunks:
{chunk_list}

For each chunk, judge whether it contains information relevant to answering 
the question. Reply with a JSON object in exactly this format:
{{"scores": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]}}

Use 1 for relevant, 0 for not relevant. Include one score per chunk in order."""

    try:
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        scores = data.get("scores", [])
        if not scores:
            return 0.0
        # Trim or pad to match chunk count
        scores = scores[:len(chunks)]
        return round(sum(scores) / len(chunks), 3)
    except Exception as e:
        log.warning(f"Context precision scoring failed: {e}")
        return 0.0


# Metric 2: Answer faithfulness


@retry(wait=WAIT)
def score_answer_faithfulness(
    answer: str, chunks: list[Result]
) -> float:
    """
    Score whether the answer is grounded in the retrieved context.
    Returns 0-1: fraction of answer claims supported by the context.
    """
    if not chunks or not answer:
        return 0.0

    context = "\n\n".join(c.page_content[:400] for c in chunks[:5])

    prompt = f"""You are evaluating whether an AI answer is faithful to its source context.

Context from SEC filings:
{context}

Answer to evaluate:
{answer}

Break the answer into individual factual claims. For each claim, judge whether 
it is directly supported by the context above.

Reply with a JSON object in exactly this format:
{{"total_claims": 5, "supported_claims": 4, "faithfulness": 0.8}}

Be strict: a claim is only supported if the context explicitly contains that information."""

    try:
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return round(float(data.get("faithfulness", 0.0)), 3)
    except Exception as e:
        log.warning(f"Faithfulness scoring failed: {e}")
        return 0.0


# Metric 3: Answer relevance


@retry(wait=WAIT)
def score_answer_relevance(question: str, answer: str) -> float:
    """
    Score whether the answer actually addresses the question asked.
    Returns 0-1.
    """
    if not answer:
        return 0.0

    prompt = f"""You are evaluating whether an AI answer addresses the question asked.

Question: {question}

Answer: {answer}

Score how well the answer addresses the question on a scale of 0 to 1:
- 1.0: directly and completely answers the question
- 0.7: mostly answers the question with minor gaps
- 0.4: partially answers the question
- 0.1: tangentially related but doesn't answer the question
- 0.0: does not address the question at all

Reply with a JSON object in exactly this format:
{{"relevance": 0.8, "reason": "one sentence explanation"}}"""

    try:
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return round(float(data.get("relevance", 0.0)), 3)
    except Exception as e:
        log.warning(f"Answer relevance scoring failed: {e}")
        return 0.0


# Evaluate a single sample


def evaluate_sample(sample: EvalSample) -> EvalResult:
    """Run all three metrics on a single eval sample."""
    log.info(f"Evaluating: {sample.question[:60]}...")

    context_precision = score_context_precision(sample.question, sample.chunks)
    answer_faithfulness = score_answer_faithfulness(sample.answer, sample.chunks)
    answer_relevance = score_answer_relevance(sample.question, sample.answer)

    result = EvalResult(
        question=sample.question,
        context_precision=context_precision,
        answer_faithfulness=answer_faithfulness,
        answer_relevance=answer_relevance,
    )

    log.info(
        f"Scores — precision: {context_precision:.2f} | "
        f"faithfulness: {answer_faithfulness:.2f} | "
        f"relevance: {answer_relevance:.2f} | "
        f"overall: {result.overall:.2f}"
    )
    return result


# Run full evaluation suite


def run_evaluation(
    questions: list[str],
    ticker: str | None = None,
) -> list[EvalResult]:
    """
    Run the full evaluation suite over a list of questions.

    Args:
        questions:  List of evaluation questions
        ticker:     Optional ticker to restrict retrieval

    Returns:
        List of EvalResult objects
    """
    results: list[EvalResult] = []

    for question in questions:
        try:
            answer, chunks = answer_question(question, ticker=ticker)
            sample = EvalSample(
                question=question,
                answer=answer,
                chunks=chunks,
            )
            result = evaluate_sample(sample)
            results.append(result)
        except Exception as e:
            log.error(f"Evaluation failed for question '{question[:50]}': {e}")

    return results


def print_eval_report(results: list[EvalResult]) -> None:
    """Print a formatted evaluation report to stdout."""
    if not results:
        print("\nNo evaluation results.\n")
        return

    print("\n" + "=" * 70)
    print("RAG EVALUATION REPORT")
    print("=" * 70)
    print(f"{'Question':<45} {'Prec':>6} {'Faith':>6} {'Rel':>6} {'Avg':>6}")
    print("-" * 70)

    for r in results:
        q = r.question[:43] + ".." if len(r.question) > 45 else r.question
        print(
            f"{q:<45} {r.context_precision:>6.2f} "
            f"{r.answer_faithfulness:>6.2f} "
            f"{r.answer_relevance:>6.2f} "
            f"{r.overall:>6.2f}"
        )

    print("-" * 70)
    avg_precision = sum(r.context_precision for r in results) / len(results)
    avg_faithfulness = sum(r.answer_faithfulness for r in results) / len(results)
    avg_relevance = sum(r.answer_relevance for r in results) / len(results)
    avg_overall = sum(r.overall for r in results) / len(results)

    print(
        f"{'AVERAGE':<45} {avg_precision:>6.2f} "
        f"{avg_faithfulness:>6.2f} "
        f"{avg_relevance:>6.2f} "
        f"{avg_overall:>6.2f}"
    )
    print("=" * 70)
    print()


# CLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline quality")
    parser.add_argument(
        "--ticker", default=None,
        help="Restrict evaluation to a specific ticker"
    )
    parser.add_argument(
        "--questions", nargs="+", default=None,
        help="Custom evaluation questions (uses defaults if not provided)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Save results to a JSON file e.g. eval_results.json"
    )
    args = parser.parse_args()

    questions = args.questions or DEFAULT_QUESTIONS
    if args.ticker:
        questions = [
            q.replace("the company", args.ticker)
            for q in questions
        ]

    log.info(f"Running evaluation — {len(questions)} question(s)")
    results = run_evaluation(questions, ticker=args.ticker)
    print_eval_report(results)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(
                [r.model_dump() for r in results],
                f,
                indent=2,
            )
        log.info(f"Results saved to {output_path}")