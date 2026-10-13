import os
import json
import time
import http.client
import urllib.parse
from typing import Dict, List, Optional


SERPER_HOST = "google.serper.dev"


class SerperClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_sec: float = 30.0,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0,
        logger=None,
    ):
        self.api_key = api_key or os.environ.get("SERPER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "SerperClient requires an API key (constructor arg or SERPER_API_KEY env var)."
            )
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.n_calls = 0
        self.logger = logger

    def search(
        self,
        query: str,
        num_results: int = 5,
        gl: str = "us",
        hl: str = "en",
        log_context: Optional[Dict] = None,
    ) -> List[str]:
        if not query or not query.strip():
            return []

        payload = json.dumps({
            "q": query.strip(),
            "num": max(1, min(num_results, 10)),
            "gl": gl,
            "hl": hl,
        })
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            conn = None
            started = time.perf_counter()
            status = None
            error = None
            snippets = []
            try:
                conn = http.client.HTTPSConnection(SERPER_HOST, timeout=self.timeout_sec)
                conn.request("POST", "/search", payload, headers)
                resp = conn.getresponse()
                status = resp.status
                raw = resp.read().decode("utf-8", errors="replace")
                conn.close()
                self.n_calls += 1

                if status == 200:
                    data = json.loads(raw)
                    snippets = self._format_snippets(data, num_results)
                    if self.logger is not None:
                        self.logger.log_search({
                            **(log_context or {}), "query": query.strip(),
                            "query_sha256": self._hash_query(query),
                            "attempt": attempt, "status": status,
                            "latency_sec": round(time.perf_counter() - started, 6),
                            "n_results": len(snippets), "failure": None,
                        })
                    return snippets

                if status < 500 and status != 429:
                    error = raw[:500]
                    print(f"[serper] non-retryable HTTP {status}: {raw[:200]}")
                    if self.logger is not None:
                        self.logger.log_search({
                            **(log_context or {}), "query": query.strip(),
                            "query_sha256": self._hash_query(query),
                            "attempt": attempt, "status": status,
                            "latency_sec": round(time.perf_counter() - started, 6),
                            "n_results": 0, "failure": error,
                        })
                    return []
                last_error = f"HTTP {status}: {raw[:200]}"

            except (http.client.HTTPException, OSError, json.JSONDecodeError) as e:
                last_error = f"{type(e).__name__}: {e}"
                error = last_error
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

            if self.logger is not None:
                self.logger.log_search({
                    **(log_context or {}), "query": query.strip(),
                    "query_sha256": self._hash_query(query),
                    "attempt": attempt, "status": status,
                    "latency_sec": round(time.perf_counter() - started, 6),
                    "n_results": 0, "failure": error or last_error,
                })

            if attempt < self.max_retries:
                time.sleep(self.retry_backoff_sec * attempt)

        print(f"[serper] giving up after {self.max_retries} attempts: {last_error}")
        return []

    @staticmethod
    def _hash_query(query: str) -> str:
        import hashlib
        return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def _format_snippets(data: dict, num_results: int) -> List[str]:
        out = []

        ab = data.get("answerBox") or {}
        if ab.get("snippet"):
            title = ab.get("title") or "Answer"
            link = ab.get("link") or ""
            out.append(f"[Answer Box] {title}: {ab['snippet']}" + (f" ({link})" if link else ""))

        kg = data.get("knowledgeGraph") or {}
        if kg.get("description"):
            title = kg.get("title") or "Knowledge"
            out.append(f"[Knowledge] {title}: {kg['description']}")

        for item in (data.get("organic") or []):
            snippet = (item.get("snippet") or "").strip()
            if not snippet:
                continue
            title = (item.get("title") or "").strip()
            link = (item.get("link") or "").strip()
            entry = f"{title}: {snippet}" if title else snippet
            if link:
                entry += f" ({link})"
            out.append(entry)
            if len(out) >= num_results:
                break

        return out[:num_results]


_DEFAULT_CLIENT: Optional[SerperClient] = None


def get_default_client() -> SerperClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = SerperClient()
    return _DEFAULT_CLIENT


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m search.serper_client \"<search query>\"")
        print("external service point: requires the SERPER_API_KEY environment variable")
        raise SystemExit(2)
    q = sys.argv[1]
    client = SerperClient()
    results = client.search(q, num_results=5)
    print(f"Found {len(results)} snippets:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r[:200]}")
