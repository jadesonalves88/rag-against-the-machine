"""Python Fire commands and configurable JSON file boundaries."""

import json
import sys
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from src.evaluation import evaluate_results
from src.generation import Generator
from src.models import (
    MinimalAnswer, MinimalSearchResults, RagDataset, StudentSearchResults,
    StudentSearchResultsAndAnswer, UnansweredQuestion,
)
from src.retrieval import Retriever, build_index


def save_json(path: Path, value: BaseModel) -> str:
    """Write validated JSON through a temporary file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return str(path)


def load_results(path: str) -> StudentSearchResults:
    """Validate a JSON search-results file before using its sources."""
    return StudentSearchResults.model_validate_json(
        Path(path).read_text(encoding="utf-8"))


class CLI:
    """Index, retrieve, answer, and evaluate a local source-code corpus."""

    def index(
        self, max_chunk_size: int = 2000, raw_directory: str = "data/raw",
        index_directory: str = "data/processed",
    ) -> str:
        """Ingest raw files and persist a BM25 index."""
        count, seconds = build_index(raw_directory, index_directory,
                                     max_chunk_size)
        return (f"Ingestion complete! {count} chunks in {seconds:.2f}s. "
                f"Indices saved under {index_directory}/")

    def search(
        self, query: str, k: int = 5,
        index_directory: str = "data/processed",
        output_path: str = "data/output/search_results/single.json",
    ) -> str:
        """Retrieve one question and save StudentSearchResults JSON."""
        question = UnansweredQuestion(question=query)
        found = Retriever(index_directory).search(query, k)
        result = MinimalSearchResults(
            **question.model_dump(), retrieved_sources=found)
        return save_json(Path(output_path), StudentSearchResults(
            search_results=[result], k=k))

    def search_dataset(
        self, dataset_path: str, k: int = 5,
        save_directory: str = "data/output/search_results",
        index_directory: str = "data/processed",
    ) -> str:
        """Retrieve a JSON question batch, preserving question identifiers."""
        if k < 0:
            raise ValueError("k must be nonnegative")
        dataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8"))
        retriever = Retriever(index_directory)
        results = [MinimalSearchResults(
            question_id=question.question_id, question=question.question,
            retrieved_sources=retriever.search(question.question, k),
        ) for question in tqdm(dataset.rag_questions, desc="Searching",
                               file=sys.stderr)]
        return save_json(Path(save_directory) / Path(dataset_path).name,
                         StudentSearchResults(search_results=results, k=k))

    def answer(
        self, query: str, k: int = 5,
        index_directory: str = "data/processed",
        output_path: str = "data/output/search_results_and_answer/single.json",
        model_name: str = "Qwen/Qwen3-0.6B",
        context_tokens: int = 8192, max_new_tokens: int = 512,
    ) -> str:
        """Retrieve and generate a grounded answer for one question."""
        sources = Retriever(index_directory).search(query, k)
        question = UnansweredQuestion(question=query)
        text = "The retrieved sources do not provide enough information."
        if sources and query.strip():
            text = Generator(model_name, context_tokens,
                             max_new_tokens).answer(query, sources)
        result = MinimalAnswer(**question.model_dump(),
                               retrieved_sources=sources, answer=text)
        return save_json(Path(output_path), StudentSearchResultsAndAnswer(
            search_results=[result], k=k))

    def answer_dataset(
        self, student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        model_name: str = "Qwen/Qwen3-0.6B",
        context_tokens: int = 8192, max_new_tokens: int = 512,
    ) -> str:
        """Answer existing search results with one model load."""
        results = load_results(student_search_results_path)
        generator: Generator | None = None
        answers: list[MinimalAnswer] = []
        for result in tqdm(results.search_results, desc="Answering",
                           file=sys.stderr):
            text = "The retrieved sources do not provide enough information."
            if result.retrieved_sources and result.question.strip():
                if generator is None:
                    generator = Generator(model_name, context_tokens,
                                          max_new_tokens)
                text = generator.answer(result.question,
                                        result.retrieved_sources)
            answers.append(MinimalAnswer(**result.model_dump(), answer=text))
        return save_json(
            Path(save_directory) / Path(student_search_results_path).name,
            StudentSearchResultsAndAnswer(search_results=answers, k=results.k),
        )

    def evaluate(
        self, student_search_results_path: str, dataset_path: str,
        output_path: str = "data/output/evaluation.json",
    ) -> str:
        """Report local recall@k without importing the official evaluator."""
        results = load_results(student_search_results_path)
        dataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text(encoding="utf-8"))
        report = evaluate_results(results, dataset)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return json.dumps(report, indent=2)


def main() -> None:
    """Dispatch Fire commands and turn operational errors into clean exits."""
    import fire

    try:
        fire.Fire(CLI())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
