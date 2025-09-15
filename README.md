# Semantic Chunker
 Utility
Semantic chunking utility that converts a plain text / markdown document into coherent sections sized for RAG (Retrieval Augmented Generation) based systems and LLM processing.

## What It Does
Given a text file, the script:
1. Splits the document into paragraphs (blank line delimiter).
2. Embeds each paragraph using a Sentence-Transformers model (default `sentence-transformers/all-MiniLM-L6-v2`).
3. Computes cosine similarities between adjacent paragraphs and (optionally) smooths them.
4. Determines section boundaries where similarity drops (adaptive threshold) or when adding another paragraph would exceed the token budget.
5. Enforces a hard per-section token limit (`--max-tokens`) with a greedy paragraph packing pass and sentence-level fallback for oversized paragraphs.
6. Optionally auto-selects a smoothing window (`--auto-smooth`) based on variance of similarity scores.
7. Outputs one JSON file containing structured chunk metadata and text.

## Output Format
The JSON file is an array of objects:
```jsonc
{
  "index": 0,
  "start_paragraph": 0,
  "end_paragraph": 3,
  "paragraph_count": 4,
  "token_estimate": 497,
  "text": "Full concatenated text of the section..."
}
```
Fields:
- `index`: zero-based section number after final packing
- `start_paragraph` / `end_paragraph`: inclusive paragraph indices
- `paragraph_count`: number of paragraphs in this chunk
- `token_estimate`: token count (exact via tiktoken if available, else heuristic)
- `text`: the chunk content (paragraphs separated by blank line)

## Token Counting
By default the script uses `tiktoken` (encoding `cl100k_base`) if installed for precise counts. If unavailable or `--no-precise-tokens` is set, it falls back to a simple heuristic multiplier on whitespace tokens.

## Installation
```bash
pip install sentence-transformers tiktoken nltk
nltk.download('punkt')
```

## Basic Usage
```bash
python chunk_document_sections.py \
  --input my_report.txt \
  --output my_report_chunks.json \
  --max-tokens 1024 \
  --auto-smooth
```

## Important Options
| Option | Description |
|--------|-------------|
| `--input PATH` | Plain text / markdown source (required). |
| `--output PATH` | Output JSON file (default `segments.json`). |
| `--model NAME` | Sentence-Transformers model (default MiniLM L6 v2). |
| `--max-tokens N` | Hard token limit per chunk (default 512). |
| `--auto-smooth` | Automatically pick smoothing window (1–9, odd). |
| `--smooth-window N` | Manual smoothing window (ignored if auto). |
| `--no-precise-tokens` | Disable tiktoken; use heuristic estimator. |

## How Boundary Detection Works
1. Raw consecutive cosine similarities are computed.
2. (Optional) Moving-average smoothing applied with window W.
3. Adaptive threshold = mean(sim) - 0.35 * stdev(sim).
4. A new boundary is inserted when similarity drops below threshold or token limit would be exceeded.
5. After initial sections are built, an adaptive packing pass merges adjacent small sections without breaching `--max-tokens`.

## Sentence-Level Fallback (NLTK)
If an individual paragraph on its own exceeds the token cap it is split into sentences using `nltk.sent_tokenize`:
- Install if needed:
  ```bash
  pip install nltk
  python - <<'PY'
  import nltk
  nltk.download('punkt')
  PY
  ```
- Sentences are greedily packed into sub-chunks without crossing `--max-tokens`.
- Very long single sentences (rare) may still exceed the cap; they are emitted as-is.

## Auto Smoothing Heuristic
When `--auto-smooth` is used:
- Similarity variance drives chosen window (higher variance ⇒ smaller window, lower ⇒ larger) within 1–9 (odd integer).
- Goal: retain genuine topic shifts while suppressing noise.

## Example
```bash
python chunk_document_sections.py \
  --input sample_article_text.txt \
  --max-tokens 1024 \
  --auto-smooth \
  --output sample_chunks.json
```
Console output includes number of paragraphs, chosen window, raw boundaries count, final chunks, and total tokens.

## Exit Codes
- 0 success
- Non‑zero: dependency/import or runtime error

## License / Attribution
Uses models provided by `sentence-transformers` (see that project's licensing). Ensure compliance with any model-specific terms.

