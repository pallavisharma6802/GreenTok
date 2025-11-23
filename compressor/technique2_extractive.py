import re
from typing import List


def _split_sentences(text: str) -> List[str]:
    #sentence splitter: split on sentence-ending punctuation
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # fallback: if no punctuation, split by lines
    if len(parts) == 1:
        parts = [p for line in text.splitlines() for p in line.split('\n') if p.strip()]
    return [p.strip() for p in parts if p.strip()]


def optimize_extractive(text: str, max_sentences: int = 2, model=None) -> str:
    if not text:
        return text

    # lazy import to keep dependency optional
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            return text
    else:
        try:
            from sentence_transformers import util  # type: ignore
        except Exception:
            return text

    sentences = _split_sentences(text)
    if not sentences:
        return text

    sent_emb = model.encode(sentences, convert_to_tensor=True)
    doc_emb = model.encode([text], convert_to_tensor=True)
    sims = util.cos_sim(sent_emb, doc_emb).squeeze(-1).tolist()
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    selected_idx = sorted(ranked[:max_sentences])
    return ' '.join([sentences[i] for i in selected_idx])
