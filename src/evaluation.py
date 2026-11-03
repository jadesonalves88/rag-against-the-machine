"""Independent character-span intersection-over-union recall evaluation."""

from src.models import (
    AnsweredQuestion, MinimalSource, RagDataset, StudentSearchResults,
)


def overlaps(reference: MinimalSource, result: MinimalSource) -> bool:
    """Match exact file paths and character-span IoU of at least 0.05."""
    if reference.file_path != result.file_path:
        return False
    intersection = max(0, min(reference.last_character_index,
                              result.last_character_index)
                       - max(reference.first_character_index,
                             result.first_character_index))
    union = (reference.last_character_index - reference.first_character_index
             + result.last_character_index - result.first_character_index
             - intersection)
    return union > 0 and intersection / union >= 0.05


def evaluate_results(
    results: StudentSearchResults, dataset: RagDataset,
) -> dict[str, object]:
    """Macro-average per-question recall; missing predictions count as zero."""
    references = [q for q in dataset.rag_questions
                  if isinstance(q, AnsweredQuestion) and q.sources]
    if not references:
        raise ValueError("Evaluation needs answered questions with sources")
    predictions = {q.question_id: q for q in results.search_results}
    if len(predictions) != len(results.search_results):
        raise ValueError("Duplicate prediction question_id")
    if len({q.question_id for q in references}) != len(references):
        raise ValueError("Duplicate reference question_id")
    report: dict[str, object] = {"questions": len(references)}
    for k in sorted({min(value, results.k) for value in (1, 3, 5, 10)}
                    | {results.k}):
        total = 0.0
        for question in references:
            prediction = predictions.get(question.question_id)
            retrieved = prediction.retrieved_sources[:k] if prediction else []
            hits = sum(any(overlaps(source, found) for found in retrieved)
                       for source in question.sources)
            total += hits / len(question.sources)
        report[f"recall@{k}"] = total / len(references)
    return report
