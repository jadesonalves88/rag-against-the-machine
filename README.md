*This project has been created as part of the 42 curriculum by jaalves-.*

## Description

RAG against the machine answers questions about a local codebase. It indexes
Python and documentation files, retrieves exact source locations with BM25,
and supplies the retrieved excerpts to the local Qwen/Qwen3-0.6B model.
Python 3.10–3.13, uv, Python Fire, Pydantic and tqdm implement the mandatory
subject interface. This implementation currently targets the mandatory part.

## Instructions

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv sync
```

Place the **provided** vLLM attachment at `data/raw/vllm-0.10.1/` and the
provided question JSON files in `data/datasets/UnansweredQuestions/` and
`data/datasets/AnsweredQuestions/`. Do not substitute a different vLLM revision:
reference paths and character offsets depend on the original attachment.
The corpus and reference datasets are not bundled with this repository.

The first answer command downloads model weights from Hugging Face. Subsequent
runs can use its local cache. Allow several GB of disk space and sufficient RAM
for CPU inference (about 3 GB for FP32 model weights, plus runtime memory);
installation does not itself download model weights.
Linux uses CPU PyTorch wheels to avoid downloading a CUDA runtime.
Set `HF_HOME` to a location with enough free disk space before running answer
commands. Choose persistent storage if the cached model must survive cleanup or reboot.

## Example usage

```sh
uv run python -m src index --max_chunk_size 2000
uv run python -m src search 'How do I enable LoRA?' --k 5
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 \
  --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
uv run python -m src answer 'How do I enable LoRA?' --k 5
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Use separate output directories for AnsweredQuestions and UnansweredQuestions,
which can contain identically named files. Batch commands preserve input
filenames and question IDs. Single-query commands save one-item batch JSON.
All paths are configurable: `index` accepts `--raw_directory` and
`--index_directory`; retrieval accepts `--index_directory`; single-query and
evaluation commands accept `--output_path`; batch commands accept
`--save_directory`. Run from the repository root. Corpus files must reside
inside that root, since source paths must be project-relative.

Generation accepts `--model_name` (default `Qwen/Qwen3-0.6B`),
`--context_tokens` (default 8192), and `--max_new_tokens` (default 512).
`uv run python -m src --help` lists commands; append `--help` to a command
for its arguments. Errors produce a short stderr message and a nonzero exit
status. Blank and out-of-vocabulary queries, and `k=0`, return empty source
lists; negative k is rejected. No-source answers abstain without loading weights.

```sh
make install
make run ARGS="search 'LoRA adapters' --k 5"
make debug ARGS="index --max_chunk_size 2000"
make lint
make clean
```

`clean` removes development caches; it preserves the corpus and generated index.

## System architecture

1. `src/chunking.py` segments original file contents without normalizing text.
2. `src/retrieval.py` builds sparse BM25 weights and persists `bm25.npz` with
   Pydantic-validated `index.json` metadata under `data/processed/`.
3. A retriever loads the index once per command and ranks all batch questions.
4. `src/generation.py` reads source ranges and packs excerpts into the available
   token budget, reserving output tokens and the chat-template overhead.
5. `src/cli.py` validates JSON boundaries and writes the subject's search or
   search-and-answer models. `src/evaluation.py` computes local recall.

Indexing and batch commands display tqdm progress. Search never loads the LLM.
The model is loaded lazily once for a batch, evaluated on CPU with gradients
disabled, FP32 weights for CPU compatibility, and at most four compute threads. It is
prompted to cite excerpts and acknowledge insufficient evidence.
Thinking is disabled through the Qwen chat template. A generated answer remains
ordinary text inside a validated JSON envelope; model-generated JSON is not
trusted for the output structure.

## Chunking strategy

Python files use AST class/function boundaries, including decorators. Enclosing
symbol names are retained as retrieval metadata. Invalid Python falls back to
bounded text windows. Markdown, reStructuredText and plain text use headings
and paragraph boundaries. Small adjacent regions are packed together.
Oversized regions use line-aware windows with up to 200 characters of overlap
(20% of the configured size for small windows). Even a single long line is
split within the limit. Whitespace-only chunks are omitted.

The default and largest allowed chunk size is 2000 **characters**, not bytes or
tokens. Offsets are half-open `[first_character_index, last_character_index)`.
UTF-8 files are read with newline translation disabled, preserving Unicode and
CRLF offsets. Every retrieved location uses the original project-relative path.
Supported extensions are `.py`, `.md`, `.mdx`, `.rst` and `.txt`; binary,
unreadable and non-UTF-8 files are skipped with diagnostics. Symlink files and
common environment/cache directories are excluded.

## Retrieval method

BM25 uses `k1=1.5`, `b=0.75` and positive Robertson IDF:
`log(1 + (N - df + 0.5) / (df + 0.5))`. Full identifiers remain searchable;
snake_case and CamelCase components are also indexed. File paths and enclosing
Python symbols contribute twice to term counts. Query term weights are
`1 + log(query_frequency)`. Positive BM25 scores determine the final ranking;
unmatched queries return no fabricated sources. A compressed sparse column
matrix supports fast term lookup, and partial selection avoids sorting the
entire corpus per query.

## Performance analysis

Measured on 2026-10-23 with an Intel Core i5-7360U (2 cores / 4 threads),
Python 3.13.1, on this local CPU environment, using the supplied
vLLM 0.10.1 archive and public datasets (100 docs questions and 99 code
questions). The default 2000-character configuration was validated by the
provided Ubuntu moulinette, run separately from the submitted implementation.
Both outputs were reported as valid with `max_context_length=2000`.

| Public dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required recall@5 |
| --- | --- | --- | --- | --- | --- |
| Docs (100) | 59.00% | 79.00% | **82.00%** | 89.00% | >= 80% |
| Code (99) | 63.64% | 82.83% | **89.90%** | 92.93% | >= 50% |

Indexing 1,969 supported files produced 14,184 chunks in **36.62 seconds**,
below the 300-second limit. A single CLI `search_dataset` run with 200 questions
took **2.04 seconds**, including process startup, loading the index and writing
JSON, below the 90-second limit. This timing batch contains all 199 public
questions and one repeated docs query with a distinct ID. It is a throughput
measurement, not an additional quality dataset. Four separate search runs over
the docs/code datasets in both question scopes produced identical recall.

The following chunk-size experiment used the same corpus, unchanged ranking,
and the independent local evaluator. Each size was indexed from scratch.
The default configuration remains 2000 characters as required by the subject.

| Maximum characters | Chunks | Index seconds | Docs recall@5 | Code recall@5 |
| --- | --- | --- | --- | --- |
| 1000 | 28,869 | 31.90 | 80.00% | 86.87% |
| 1500 | 18,919 | 36.13 | 83.00% | 88.89% |
| 2000 (default) | 14,184 | 36.62 | 82.00% | 89.90% |

Compared with 2000 characters, 1000-character chunks reduced docs recall by
2 percentage points and code recall by 3.03 points. At 1500, docs improved by
1 point while code decreased by 1.01 points. The observations support retaining
the required default; larger context helped code retrieval on this public set.
Index times vary with system load and filesystem caches; fewer chunks did not
consistently imply a shorter indexing run.

The local evaluator matches question IDs, requires exact file paths and span
intersection-over-union >= 0.05, and macro-averages the fraction of reference
sources found per answered question. Missing predictions count as zero;
questions without reference sources are excluded. It reports available cutoffs
among 1, 3, 5 and 10 plus the requested k. Its default-configuration scores
matched the official executable on these public datasets. The implementation
never imports or invokes that executable. Public scores do not guarantee the
same recall on unseen evaluation questions or the same timings on other CPUs.

Generation was checked with the real Qwen/Qwen3-0.6B model using a docs
question about `/v1/load_lora_adapter` and a code question about fused batched
MoE activation formats. Both the single-query command and a two-question
`answer_dataset` run produced Pydantic-valid JSON. The batch used a
4096-token context budget and up to 96 generated tokens. Its answers identified
the expected endpoint and `FusedMoEActivationFormat.BatchedExperts`; the code
answer omitted the tuple-return detail, illustrating the model's permitted
partial incompleteness. This was a generation smoke test, not an evaluation of
answers for all 199 questions. CPU generation is substantially slower than
lexical retrieval.

## Design decisions

Sparse lexical retrieval is CPU-friendly and needs no embedding downloads.
JSON metadata avoids loading executable pickle files. Pydantic models validate
stage boundaries and retrieved source lengths. The reference-source model
allows spans longer than 2000; the size cap applies to retrieved context.
Generation budgets the complete tokenized prompt rather than assuming a fixed
characters-to-tokens ratio. Search and generation are separate so retrieval
quality can be evaluated without model inference.

## Challenges faced

Natural-language questions and code identifiers use different vocabulary;
identifier splitting and path/symbol metadata address part of that gap.
Exact path prefixes and Unicode/newline offsets are essential to grading, so
source text is never normalized before calculating ranges. The supplied public
datasets now verify the retrieval thresholds; unseen questions still need
independent evaluation. Model grounding and answer quality also require human
review, beyond JSON schema validation.

## Resources

- [Okapi at TREC-7](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/okapi_trec7.pdf): BM25 retrieval.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B): model and chat-template usage.
- [uv documentation](https://docs.astral.sh/uv/): dependency and environment management.
- [Python Fire](https://github.com/google/python-fire): command-line interface.
- [Pydantic](https://docs.pydantic.dev/latest/): validated exchange models.
- [SciPy sparse matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html): lexical index storage.
- [Python AST](https://docs.python.org/3/library/ast.html): Python segmentation.

AI assistance was used to read the subject, design and implement the initial
pipeline, prepare documentation, and develop and run validation checks.
The student must review and understand every submitted component, reproduce
the measurements, and seek peer review before defense.
