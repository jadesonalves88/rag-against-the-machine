"""Validated exchange formats required by the subject."""

import uuid
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator


class MinimalSource(BaseModel):
    """A half-open character range in a corpus file."""

    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "MinimalSource":
        """Reject empty or reversed source ranges."""
        width = self.last_character_index - self.first_character_index
        if width <= 0:
            raise ValueError("Source ranges must have positive width")
        return self


class UnansweredQuestion(BaseModel):
    """A question with a stable identifier."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A reference question, answer, and ground-truth sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A collection of answered or unanswered questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]

    @field_validator("rag_questions", mode="before")
    @classmethod
    def validate_questions(cls, value: object) -> object:
        """Keep malformed reference sources from becoming unanswered items."""
        if not isinstance(value, list):
            return value
        return [AnsweredQuestion.model_validate(item)
                if isinstance(item, dict) and "sources" in item else item
                for item in value]


class MinimalSearchResults(BaseModel):
    """Ranked source locations for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

    @model_validator(mode="after")
    def validate_context_lengths(self) -> "MinimalSearchResults":
        """Enforce the evaluator's maximum retrieved source width."""
        if any(s.last_character_index - s.first_character_index > 2000
               for s in self.retrieved_sources):
            raise ValueError("Retrieved sources cannot exceed 2000 characters")
        return self


class MinimalAnswer(MinimalSearchResults):
    """Search results augmented with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Batch retrieval output."""

    search_results: List[MinimalSearchResults]
    k: int = Field(ge=0)


class StudentSearchResultsAndAnswer(BaseModel):
    """Batch generation output."""

    search_results: List[MinimalAnswer]
    k: int = Field(ge=0)


class Chunk(MinimalSource):
    """An indexed source range and its exact original text."""

    text: str
    symbols: str = ""

    @model_validator(mode="after")
    def validate_text(self) -> "Chunk":
        """Verify persisted chunk text agrees with its bounded range."""
        width = self.last_character_index - self.first_character_index
        if width != len(self.text) or width > 2000:
            raise ValueError("Chunk text must match its bounded source range")
        return self


class IndexManifest(BaseModel):
    """Metadata needed to load a persisted lexical index."""

    version: int = 1
    max_chunk_size: int
    vocabulary: dict[str, int]
    chunks: List[Chunk]
