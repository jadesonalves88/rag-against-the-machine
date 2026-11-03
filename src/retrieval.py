"""Sparse BM25 indexing with identifier and path-aware tokenization."""

import math
import re
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import sparse
from tqdm import tqdm

from src.chunking import chunk_file
from src.models import Chunk, IndexManifest, MinimalSource

EXTENSIONS = {".py", ".md", ".mdx", ".rst", ".txt"}
EXCLUDED = {".git", ".venv", "__pycache__", "node_modules"}


def tokenize(text: str) -> list[str]:
    """Retain full identifiers and add snake_case/CamelCase components."""
    tokens: list[str] = []
    for word in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+", text):
        lower = word.lower()
        tokens.append(lower)
        parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word)
        parts = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", parts)
        split = re.findall(r"[a-z0-9]+", parts.lower())
        if split != [lower]:
            tokens.extend(split)
    return tokens


def build_index(
    raw_directory: str, index_directory: str, max_chunk_size: int,
) -> tuple[int, float]:
    """Read UTF-8 sources and persist BM25 weights and validated metadata."""
    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("max_chunk_size must be between 1 and 2000")
    started = time.perf_counter()
    root = Path(raw_directory)
    if not root.is_dir():
        raise ValueError(f"Corpus directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*")
                   if path.is_file() and path.suffix.lower() in EXTENSIONS
                   and not EXCLUDED.intersection(path.parts)
                   and not path.is_symlink())
    chunks: list[Chunk] = []
    for path in tqdm(files, desc="Indexing", file=sys.stderr):
        try:
            # newline='' preserves CRLF and therefore character offsets.
            with path.open(encoding="utf-8", newline="") as stream:
                text = stream.read()
            if "\x00" in text:
                continue
            relative = path.resolve().relative_to(Path.cwd().resolve())
            chunks.extend(chunk_file(text, relative.as_posix(),
                                     max_chunk_size, path.suffix == ".py"))
        except (OSError, UnicodeError) as error:
            print(f"Skipping {path}: {error}", file=sys.stderr)
        except ValueError as error:
            raise ValueError(
                "Corpus must be inside the project root") from error
    if not chunks:
        raise ValueError("No nonempty supported source files found")
    vocabulary: dict[str, int] = {}
    rows: list[int] = []
    columns: list[int] = []
    counts: list[float] = []
    lengths: list[int] = []
    for row, chunk in enumerate(tqdm(chunks, desc="Building BM25")):
        words = tokenize(chunk.text)
        # Path and enclosing symbols help natural-language/identifier matching.
        words += tokenize(chunk.file_path) * 2 + tokenize(chunk.symbols) * 2
        lengths.append(len(words))
        frequencies: dict[int, int] = {}
        for word in words:
            column = vocabulary.setdefault(word, len(vocabulary))
            frequencies[column] = frequencies.get(column, 0) + 1
        for column, frequency in frequencies.items():
            rows.append(row)
            columns.append(column)
            counts.append(float(frequency))
    matrix: Any = sparse.csr_matrix(
        (np.asarray(counts, dtype=np.float32), (rows, columns)),
        shape=(len(chunks), len(vocabulary)),
        dtype=np.float32,
    )
    average = max(float(np.mean(lengths)), 1.0)
    norms = 1.5 * (0.25 + 0.75 * np.asarray(lengths) / average)
    matrix.data = (matrix.data * 2.5
                   / (matrix.data + np.repeat(norms, np.diff(matrix.indptr))))
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    idf = np.log1p(
        (len(chunks) - document_frequency + 0.5) / (document_frequency + 0.5)
    )
    matrix = matrix.multiply(idf).tocsc()
    output = Path(index_directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest = IndexManifest(max_chunk_size=max_chunk_size,
                             vocabulary=vocabulary, chunks=chunks)
    # Store text and metadata as JSON, and BM25 weights as sparse arrays.
    sparse.save_npz(output / "bm25.npz", matrix)
    (output / "index.json").write_text(manifest.model_dump_json(),
                                       encoding="utf-8")
    return len(chunks), time.perf_counter() - started


class Retriever:
    """Load a lexical index once and efficiently rank multiple queries."""

    def __init__(self, index_directory: str) -> None:
        """Read index metadata and its sparse scoring matrix."""
        directory = Path(index_directory)
        self.manifest = IndexManifest.model_validate_json(
            (directory / "index.json").read_text(encoding="utf-8"))
        if self.manifest.version != 1:
            raise ValueError("Unsupported index version; rebuild the index")
        self.matrix: Any = cast(Any, sparse.load_npz(
            directory / "bm25.npz")).tocsc()
        if self.matrix.shape != (len(self.manifest.chunks),
                                 len(self.manifest.vocabulary)):
            raise ValueError("Inconsistent index; run index again")

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return up to k positive-scoring, ranked source locations."""
        if k < 0:
            raise ValueError("k must be nonnegative")
        if not query.strip() or k == 0:
            return []
        frequencies: dict[int, float] = {}
        for token in tokenize(query):
            column = self.manifest.vocabulary.get(token)
            if column is not None:
                frequencies[column] = frequencies.get(column, 0.0) + 1.0
        if not frequencies:
            return []
        columns = list(frequencies)
        weights = np.array([1 + math.log(frequencies[c]) for c in columns])
        scores = np.asarray(self.matrix[:, columns] @ weights).ravel()
        positive = np.flatnonzero(scores > 0)
        count = min(k, len(positive))
        if not count:
            return []
        candidates = positive[np.argpartition(
            -scores[positive], count - 1)[:count]]
        ranked = sorted(candidates, key=lambda i: (-scores[i], int(i)))
        return [MinimalSource.model_validate(
            self.manifest.chunks[int(i)].model_dump()) for i in ranked]
