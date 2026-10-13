import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi


_WORD_RE = re.compile(r"[a-zA-Z0-9]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    tokens = _WORD_RE.findall(text)
    tokens.extend(_CJK_RE.findall(text))
    return tokens


class BM25Retriever:

    def __init__(self, evidence_pool: List[str]):
        self.pool = list(evidence_pool)
        cleaned = [(i, s) for i, s in enumerate(self.pool) if s and s.strip()]
        self._idx_map = [i for i, _ in cleaned]
        corpus_tokens = [_tokenize(s) for _, s in cleaned]

        if not corpus_tokens:
            self._bm25 = None
        else:
            self._bm25 = BM25Okapi(corpus_tokens)

    def __len__(self) -> int:
        return len(self._idx_map)

    def top_k(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self._bm25 is None or not self._idx_map:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return [(self.pool[i], 0.0) for i in self._idx_map[:k]]

        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(
            zip(self._idx_map, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )[:k]
        return [(self.pool[i], float(s)) for (i, s) in ranked]

    def top_k_sentences(self, query: str, k: int = 5) -> List[str]:
        return [s for (s, _) in self.top_k(query, k)]


if __name__ == "__main__":
    pool = [
        "Apple introduced power management features in iOS 10.2.1 to prevent unexpected shutdowns on older iPhones with degraded batteries.",
        "Apple agreed to pay up to $500 million to settle a lawsuit accusing it of slowing older iPhones.",
        "There is no Harvard study on Apple's iPhone slowdown.",
        "The Texas Insurance Code took effect on September 1, 2017.",
        "A 2018 paper from MIT analyzed iPhone performance over time.",
    ]
    r = BM25Retriever(pool)
    q = "Did Apple slow down older iPhones to boost new sales?"
    print(f"Query: {q}")
    for sent, score in r.top_k(q, k=3):
        print(f"  [{score:.3f}] {sent}")
    print()
    q2 = "Does a Harvard study exist?"
    print(f"Query: {q2}")
    for sent, score in r.top_k(q2, k=3):
        print(f"  [{score:.3f}] {sent}")
