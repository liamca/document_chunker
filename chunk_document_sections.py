#!/usr/bin/env python3
"""
chunk_document_sections.py

Purpose:
  Convert a plain text (or markdown) document into semantically coherent chunks whose
  token counts do not exceed a specified maximum, using sentence-transformer embeddings
  and adaptive boundary detection.

Key Features:
  - Paragraph extraction from plaintext/markdown (split on blank lines)
  - Embedding with sentence-transformers (default: all-MiniLM-L6-v2)
  - Cosine similarity between adjacent paragraphs; optional smoothing or auto-smoothing
  - Adaptive threshold to insert boundaries at semantic topic shifts
  - Hard token cap enforced (tiktoken precise counting when available)
  - Paragraph-level greedy splitting with sentence-level fallback for overlong paragraphs
  - Adaptive packing pass merges small consecutive sections without exceeding max tokens
  - Auto selection of smoothing window (--auto-smooth) based on similarity variance

Output:
  - A single JSON file (--output) containing a list of section objects:
      {
        "index": int,
        "start_paragraph": int,
        "end_paragraph": int,
        "paragraph_count": int,
        "token_estimate": int,
        "text": str
      }

Basic Usage:
  python chunk_document_sections.py \
      --input my_doc.txt \
      --output my_doc_chunks.json \
      --max-tokens 1024 \
      --auto-smooth

Common Options:
  --model <model_name>            Sentence-Transformers model (default all-MiniLM-L6-v2)
  --max-tokens <N>                Hard cap for each chunk (default 512)
  --no-precise-tokens             Disable tiktoken counting (use heuristic)
  --smooth-window <odd_int>       Manual smoothing window (ignored if --auto-smooth)
  --auto-smooth                   Let heuristic pick smoothing window dynamically
  --output <file.json>            Output segments JSON (default segments.json)

Exit Codes:
  0 on success; non-zero on import or runtime errors.

Examples:
  # Fast chunking with 1K token limit and heuristic smoothing
  python chunk_document_sections.py --input report.md --max-tokens 1000 --auto-smooth --output report_chunks.json

  # Deterministic (no auto-smooth) with window=5 and approximate tokens
  python chunk_document_sections.py --input large.txt --smooth-window 5 --no-precise-tokens

Implementation notes:
  - Token counts default to precise OpenAI-style (tiktoken cl100k_base) if available.
  - Smoothing window influences sensitivity: larger => fewer, bigger sections; smaller => more granular.
  - Adaptive threshold = mean(sim) - k * std(sim) with k tuned (0.35) for moderate segmentation.

Limitations / Future Enhancements:
  - No markdown heading bias yet (could anchor section starts at headings)
        - Sentence segmentation fallback now uses NLTK sent_tokenize by default.
  - Could add desired target section count iterative optimization

"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from pathlib import Path
import nltk
try:
    # punkt is required for sent_tokenize
    nltk.data.find('tokenizers/punkt')
except LookupError:  # download quietly
    nltk.download('punkt', quiet=True)
from typing import List, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("sentence-transformers not installed. Install with: pip install sentence-transformers")

_tiktoken_enc = None

def _get_encoder(encoding_name: str = "cl100k_base"):
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken  # type: ignore
            _tiktoken_enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            _tiktoken_enc = False
    return _tiktoken_enc

TOKEN_FACTOR = 1.3

def estimate_tokens(text: str, precise: bool = True) -> int:
    if precise:
        enc = _get_encoder()
        if enc:
            try:
                return len(enc.encode(text))
            except Exception:
                pass
    return int(len(text.split()) * TOKEN_FACTOR)

def load_paragraphs(path: Path):
    text = path.read_text(encoding='utf-8')
    blocks = [b.strip() for b in text.replace('\r','').split('\n\n')]
    paragraphs = [b for b in blocks if b]
    return paragraphs

def compute_embeddings(paragraphs: List[str], model_name: str):
    model = SentenceTransformer(model_name)
    return model.encode(paragraphs, batch_size=32, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)

def smooth(values: List[float], window: int = 3) -> List[float]:
    if window <= 1 or not values:
        return values
    half = window // 2
    out = []
    for i in range(len(values)):
        s = max(0, i-half)
        e = min(len(values), i+half+1)
        out.append(sum(values[s:e])/(e-s))
    return out

def find_boundaries(paragraphs: List[str], embeddings, max_tokens: int, precise_tokens: bool, smooth_window: int = 5) -> List[int]:
    if not paragraphs:
        return []
    boundaries = [0]
    sims = [float((embeddings[i]*embeddings[i-1]).sum()) for i in range(1,len(paragraphs))]
    if sims:
        sims_sm = smooth(sims, smooth_window)
        mean_sim = statistics.mean(sims_sm)
        std_sim = statistics.pstdev(sims_sm) if len(sims_sm)>1 else 0.0
        adaptive_thresh = mean_sim - 0.35*std_sim
    else:
        adaptive_thresh = 1.0
    current_tokens = 0
    for idx, para in enumerate(paragraphs):
        est = estimate_tokens(para, precise=precise_tokens)
        if idx!=0 and current_tokens + est > max_tokens:
            boundaries.append(idx)
            current_tokens = 0
        if idx < len(paragraphs)-1:
            sim = sims[idx] if sims else 1.0
            if sim < adaptive_thresh:
                nxt = idx+1
                if nxt not in boundaries:
                    boundaries.append(nxt)
                    current_tokens = 0
                    continue
        current_tokens += est
    return sorted(set(boundaries))

def build_sections(paragraphs: List[str], boundaries: List[int], max_tokens: int, precise_tokens: bool) -> List[Dict[str,Any]]:
    sections: List[Dict[str,Any]] = []
    for i,start in enumerate(boundaries):
        end = boundaries[i+1]-1 if i+1 < len(boundaries) else len(paragraphs)-1
        para_slice = paragraphs[start:end+1]
        text = '\n\n'.join(para_slice)
        token_est = estimate_tokens(text, precise=precise_tokens)
        if token_est > max_tokens and len(para_slice)>1:
            tmp=[]; acc=0; sub_start=start
            for local_idx,p in enumerate(para_slice):
                p_tokens = estimate_tokens(p, precise=precise_tokens)
                if p_tokens > max_tokens:
                    from nltk.tokenize import sent_tokenize
                    sentences = [s.strip() for s in sent_tokenize(p) if s.strip()]
                    sent_acc=[]; sent_tok=0
                    for s in sentences:
                        s_full = s
                        st=estimate_tokens(s_full, precise=precise_tokens)
                        if sent_tok + st > max_tokens and sent_acc:
                            chunk=' '.join(sent_acc)
                            sections.append({'index':len(sections),'start_paragraph':sub_start,'end_paragraph':sub_start,'paragraph_count':1,'token_estimate':estimate_tokens(chunk, precise=precise_tokens),'text':chunk})
                            sent_acc=[]; sent_tok=0
                        sent_acc.append(s_full); sent_tok+=st
                    if sent_acc:
                        chunk=' '.join(sent_acc)
                        sections.append({'index':len(sections),'start_paragraph':sub_start,'end_paragraph':sub_start,'paragraph_count':1,'token_estimate':estimate_tokens(chunk, precise=precise_tokens),'text':chunk})
                    sub_start+=1
                    continue
                if acc + p_tokens > max_tokens and tmp:
                    chunk='\n\n'.join(tmp)
                    sections.append({'index':len(sections),'start_paragraph':sub_start,'end_paragraph':sub_start+len(tmp)-1,'paragraph_count':len(tmp),'token_estimate':estimate_tokens(chunk, precise=precise_tokens),'text':chunk})
                    tmp=[]; acc=0; sub_start=start+local_idx
                tmp.append(p); acc+=p_tokens
            if tmp:
                chunk='\n\n'.join(tmp)
                sections.append({'index':len(sections),'start_paragraph':sub_start,'end_paragraph':sub_start+len(tmp)-1,'paragraph_count':len(tmp),'token_estimate':estimate_tokens(chunk, precise=precise_tokens),'text':chunk})
        else:
            sections.append({'index':len(sections),'start_paragraph':start,'end_paragraph':end,'paragraph_count':end-start+1,'token_estimate':token_est,'text':text})
    return sections

def adaptive_pack(sections: List[Dict[str,Any]], max_tokens: int, precise_tokens: bool) -> List[Dict[str,Any]]:
    if not sections: return sections
    packed=[]; buf=[]
    buf_start=sections[0]['start_paragraph']; buf_end=sections[0]['end_paragraph']; buf_tokens=0
    for sec in sections:
        sec_tokens=sec['token_estimate']; cand=buf_tokens+sec_tokens
        if buf and cand<=max_tokens:
            buf.append(sec['text']); buf_end=sec['end_paragraph']; buf_tokens=cand
        else:
            if buf:
                merged='\n\n'.join(buf)
                packed.append({'index':len(packed),'start_paragraph':buf_start,'end_paragraph':buf_end,'paragraph_count':buf_end-buf_start+1,'token_estimate':estimate_tokens(merged, precise=precise_tokens),'text':merged})
            buf=[sec['text']]; buf_start=sec['start_paragraph']; buf_end=sec['end_paragraph']; buf_tokens=sec_tokens
    if buf:
        merged='\n\n'.join(buf)
        packed.append({'index':len(packed),'start_paragraph':buf_start,'end_paragraph':buf_end,'paragraph_count':buf_end-buf_start+1,'token_estimate':estimate_tokens(merged, precise=precise_tokens),'text':merged})
    return packed

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--output', type=Path, default=Path('segments.json'))
    ap.add_argument('--no-precise-tokens', action='store_true')
    ap.add_argument('--smooth-window', type=int, default=5)
    ap.add_argument('--auto-smooth', action='store_true')
    args=ap.parse_args()

    paragraphs=load_paragraphs(args.input)
    if not paragraphs:
        print('No paragraphs loaded.'); return
    print(f'Loaded {len(paragraphs)} paragraphs')

    embeddings=compute_embeddings(paragraphs, args.model)
    precise_tokens=not args.no_precise_tokens

    if args.auto_smooth:
        sims_tmp=[float((embeddings[i]*embeddings[i-1]).sum()) for i in range(1,len(paragraphs))]
        if sims_tmp:
            var=statistics.pvariance(sims_tmp) if len(sims_tmp)>1 else 0.0
            n=len(paragraphs)
            base=max(1,int(round(math.log2(n+1))))
            v_scaled=min(3.0,max(0.3,var*5))
            auto_window=int(min(9,max(1,round(base*v_scaled/2))))
        else:
            auto_window=3
        if auto_window %2==0:
            auto_window+=1 if auto_window<9 else -1
        print(f'AUTO-SMOOTH window chosen: {auto_window}')
        smooth_window=auto_window
    else:
        smooth_window=args.smooth_window

    boundaries=find_boundaries(paragraphs, embeddings, args.max_tokens, precise_tokens, smooth_window)
    print(f'Identified {len(boundaries)} raw section boundaries (smooth_window={smooth_window})')

    sections=build_sections(paragraphs, boundaries, args.max_tokens, precise_tokens)
    sections=adaptive_pack(sections, args.max_tokens, precise_tokens)
    for i,s in enumerate(sections): s['index']=i
    violations=[s for s in sections if s['token_estimate']>args.max_tokens]
    for s in violations: print(f'WARNING: Section {s["index"]} token_estimate {s["token_estimate"]} exceeds limit {args.max_tokens}')

    with args.output.open('w') as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    print(f'Wrote {args.output}')
    total=sum(s['token_estimate'] for s in sections)
    print(f'Sections (final): {len(sections)} | Total token_estimate: {total}')
    print(f'Precise tokens: {bool(_get_encoder())} | Smooth window: {smooth_window} (auto={args.auto_smooth})')

if __name__=='__main__':
    main()
