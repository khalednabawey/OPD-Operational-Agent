"""
policy_rag.py
=============
TWO MODES:

  1. BUILD MODE  — run this file directly once to create the ChromaDB on disk:
        python policy_rag.py
        python policy_rag.py --json path/to/policy_summaries.json --db ./chroma_db

  2. QUERY MODE  — imported by agent_core_policy.py at runtime.
     Opens the persisted ChromaDB collection and embeds only the user's query.
     The query embedding model is loaded once per Python process by the agent
     singleton; the policy summaries are not re-embedded on each app start.

Flow
----
  First time (you, manually):
      python policy_rag.py
      → downloads all-MiniLM-L6-v2 once, embeds 5 docs, writes ./chroma_db/

  Every Streamlit process (automatic):
      PolicyRAG("policy_summaries.json")  ← imported by agent_core_policy.py
      → opens ./chroma_db/ and loads the query embedder once.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

import requests

# Keep transformers/sentence-transformers on the lightweight PyTorch path.
# TensorFlow is not needed for MiniLM embeddings and can destabilize Streamlit
# on Windows when it is imported into the app process.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
load_dotenv(".env")

try:
    import chromadb
    from chromadb import EmbeddingFunction, Documents, Embeddings
except ModuleNotFoundError:
    chromadb = None
    EmbeddingFunction = object
    Documents = list[str]
    Embeddings = list[list[float]]

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "opd_policies"
FINGERPRINT_KEY = "json_fingerprint"
# Ensure this matches the model used by embedding-server
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Both paths are relative to this file's location so they work regardless of
# where Streamlit is launched from.
_HERE = Path(__file__).parent
DEFAULT_JSON_PATH = _HERE / "policy_summaries.json"
DEFAULT_DB_DIR = _HERE / "chroma_db"


# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------

class _RemoteEmbeddingFunction(EmbeddingFunction):
    """Call a remote embedding server (HuggingFace TEI compatible)."""

    def __init__(self, api_url: str, model_name: str = DEFAULT_MODEL) -> None:
        self._api_url = api_url.rstrip("/")
        self._model_name = model_name
        # TEI usually uses /embed, whereas Infinity/OpenAI uses /v1/embeddings
        self._endpoint = f"{self._api_url}/embed" if "/v1" not in self._api_url else f"{self._api_url}/embeddings"
        print(f"[PolicyRAG] Using remote embedding server: {self._endpoint}")

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        payload = {"inputs": input} if "/embed" in self._endpoint else {
            "input": input, "model": self._model_name}
        resp = requests.post(
            self._endpoint,
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        # TEI returns a raw list of embeddings, OpenAI returns a data object
        return data if isinstance(data, list) else [item["embedding"] for item in data["data"]]


class _SentenceTransformerEF(EmbeddingFunction):
    """SentenceTransformer embedder used for DB build and query embedding."""

    def __init__(self, model_name: str = DEFAULT_MODEL, show_progress_bar: bool = False) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        print(f"[PolicyRAG] Loading SentenceTransformer '{model_name}' ...")
        self._model = SentenceTransformer(model_name)
        self._show_progress_bar = show_progress_bar
        print("[PolicyRAG] Model loaded.")

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        return self._model.encode(
            input,
            show_progress_bar=self._show_progress_bar,
        ).tolist()


class _ChromaDefaultEF(EmbeddingFunction):
    """
    Optional Chroma ONNX embedder. Kept available, but not the default query
    backend because it may download its model lazily on the first query.
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._ef = DefaultEmbeddingFunction()

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        return self._ef(input)


# ---------------------------------------------------------------------------
# Build helpers (only used when running as __main__)
# ---------------------------------------------------------------------------

def _fingerprint(policies: dict) -> str:
    raw = json.dumps(policies, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _build_corpus(policies: dict[str, Any]) -> tuple[list[str], list[str], list[dict]]:
    docs, ids, metas = [], [], []
    for name, data in policies.items():
        summary: str = data.get("summary", "")
        kpis: list[str] = data.get("related_kpis", [])
        doc = f"Policy: {name}. {summary} Relevant KPIs: {', '.join(kpis)}."
        docs.append(doc)
        ids.append(name)
        metas.append({
            "policy_name":  name,
            "summary":      summary,
            "related_kpis": "|".join(kpis),
        })
    return docs, ids, metas


def build_db(
    json_path: Path = DEFAULT_JSON_PATH,
    db_dir: Path = DEFAULT_DB_DIR,
    model_name: str = DEFAULT_MODEL,
    force: bool = False,
    chroma_host: str | None = None,
    embedding_server_url: str | None = os.getenv("EMBEDDING_SERVER_URL"),
) -> None:
    """
    Embed all policies and persist the ChromaDB collection to *db_dir*.
    Safe to re-run — skips if the JSON fingerprint hasn't changed (unless
    --force is passed).
    """
    if chromadb is None:
        raise RuntimeError(
            "chromadb is not installed. Install it with: pip install chromadb"
        )

    print(f"[build] Reading policies from: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        policies = json.load(f)

    fp = _fingerprint(policies)

    if chroma_host:
        host_parts = chroma_host.split(":")
        _host = host_parts[0]
        _port = int(host_parts[1]) if len(host_parts) > 1 else 8000
        
        client = chromadb.HttpClient(host=_host, port=_port)
        print(f"[build] Connecting to remote ChromaDB server at {_host}:{_port} for build.")
    else:
        db_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(db_dir))
        print(f"[build] Using local ChromaDB at '{db_dir}' for build.")

    existing = []
    try:
        existing = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing and not force:
            col = client.get_collection(COLLECTION_NAME)
            if col.metadata.get(FINGERPRINT_KEY) == fp and col.count() > 0:
                print(
                    f"[build] DB already up-to-date ({col.count()} policies). "
                    "Nothing to do. Use --force to rebuild."
                )
                return
            print("[build] Policy JSON changed — rebuilding collection ...")
            client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"[build] Starting with a fresh state: {e}")

    if embedding_server_url:
        ef = _RemoteEmbeddingFunction(embedding_server_url, model_name)
    else:
        ef = _SentenceTransformerEF(model_name, show_progress_bar=True)

    col = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine", FINGERPRINT_KEY: fp},
    )
    corpus, ids, metadatas = _build_corpus(policies)
    col.add(documents=corpus, ids=ids, metadatas=metadatas)

    if chroma_host:
        print(
            f"[build] ✅ {col.count()} policies embedded and saved to remote ChromaDB at {_host}:{_port}")
    else:
        print(
            f"[build] ✅ {col.count()} policies embedded and saved to local disk at '{db_dir}'")
    print("[build] You can now start Streamlit; policy documents will not be re-embedded.")


# ---------------------------------------------------------------------------
# PolicyRAG  — QUERY-ONLY class, imported by agent_core_policy.py
# ---------------------------------------------------------------------------

class PolicyRAG:
    """
    Query-only interface to the persisted ChromaDB collection.

    Requires the DB to have been built first:
        python policy_rag.py

    Runtime defaults to keyword/KPI matching, which avoids loading embedding
    models in Streamlit. Chroma backends are still available for experiments.

    Vector similarity needs a query vector, so every new user question still
    needs an embedder. That MiniLM load is what we avoid in the app path.
    """

    def __init__(
        self,
        json_path: str | Path = DEFAULT_JSON_PATH,   # kept for API compat
        persist_dir: str | Path = DEFAULT_DB_DIR,
        similarity_threshold: float = 0.30,
        n_results: int = 2,
        query_backend: str = "localhost",
        chroma_host: str | None = "localhost",
        model_name: str = DEFAULT_MODEL,
        groq_api_key: str | None = None,
        embedding_server_url: str | None = os.getenv("EMBEDDING_SERVER_URL"),
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.default_n_results = n_results
        self.query_backend = query_backend
        self._policies = self._load_policies(Path(json_path))
        db_dir = Path(persist_dir)
        self.chroma_host = chroma_host

        if query_backend == "keyword":
            print(
                f"[PolicyRAG] Loaded {len(self._policies)} policy summaries for keyword retrieval.")
            return

        if chromadb is None:
            raise RuntimeError(
                "chromadb is not installed. Install it with: pip install chromadb"
            )

        if self.chroma_host:
            host_parts = self.chroma_host.split(":")
            _host = host_parts[0]
            _port = int(host_parts[1]) if len(host_parts) > 1 else 8000
            
            self._client = chromadb.HttpClient(host=_host, port=_port)
        else:
            self._client = chromadb.PersistentClient(path=str(db_dir))

        existing = [c.name for c in self._client.list_collections()]
        if COLLECTION_NAME not in existing:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found in '{db_dir}'.\n"
                "Run:  python policy_rag.py"
            )

        if embedding_server_url:
            embedding_function = _RemoteEmbeddingFunction(
                embedding_server_url, model_name)
        elif query_backend == "sentence_transformer":
            embedding_function = _SentenceTransformerEF(model_name)
        elif query_backend == "chroma_default":
            embedding_function = _ChromaDefaultEF()
        else:
            raise ValueError(
                "query_backend must be 'keyword', 'sentence_transformer', or 'chroma_default'"
            )

        self._col = self._client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
        )

        if self.chroma_host:
            print(
                f"[PolicyRAG] Connected to remote ChromaDB at {self.chroma_host}. Count: {self._col.count()}")
        else:
            print(
                f"[PolicyRAG] Loaded {self._col.count()} policies from local disk. Count: {self._col.count()}")

    # ------------------------------------------------------------------
    def _load_policies(self, json_path: Path) -> dict[str, Any]:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize(self, text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _tokens(self, text: str) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "how", "i", "in", "is", "it", "me", "of", "on", "or", "policy",
            "the", "to", "what", "when", "with", "about", "tell", "explain",
        }
        tokens: set[str] = set()
        for token in self._normalize(text).split():
            if len(token) <= 2 or token in stopwords:
                continue
            tokens.add(token)
            if token.endswith("ies") and len(token) > 4:
                tokens.add(f"{token[:-3]}y")
            elif token.endswith("es") and len(token) > 4:
                tokens.add(token[:-2])
            elif token.endswith("s") and len(token) > 3:
                tokens.add(token[:-1])
        return tokens

    def _keyword_query(self, question: str, n_results: int) -> list[dict[str, Any]]:
        q_norm = self._normalize(question)
        q_tokens = self._tokens(question)
        scored: list[tuple[float, str, dict[str, Any]]] = []

        for name, data in self._policies.items():
            summary = data.get("summary", "")
            related_kpis = data.get("related_kpis", [])
            haystack = " ".join([name, summary, *related_kpis])
            h_tokens = self._tokens(haystack)

            overlap = len(q_tokens & h_tokens)
            score = min(0.65, overlap * 0.14)

            name_norm = self._normalize(name)
            if name_norm and (name_norm in q_norm or q_norm in name_norm):
                score += 0.35

            for kpi in related_kpis:
                kpi_tokens = self._tokens(kpi)
                kpi_norm = self._normalize(kpi)
                if kpi_norm and (kpi_norm in q_norm or bool(kpi_tokens & q_tokens)):
                    score += 0.20

            name_overlap = self._tokens(name) & q_tokens
            if name_overlap:
                score += 0.25

            score = round(min(score, 1.0), 4)
            if score >= self.similarity_threshold:
                scored.append((score, name, data))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "policy_name": name,
                "summary": data.get("summary", ""),
                "related_kpis": data.get("related_kpis", []),
                "similarity": score,
            }
            for score, name, data in scored[:n_results]
        ]

    def query(self, question: str, n_results: int | None = None) -> list[dict[str, Any]]:
        if self.query_backend == "keyword":
            return self._keyword_query(question, n_results or self.default_n_results)

        n = min(n_results or self.default_n_results, self._col.count())
        raw = self._col.query(query_texts=[question], n_results=n)

        results = []
        for pid, meta, dist in zip(
            raw["ids"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            similarity = round(1.0 - dist, 4)
            if similarity >= self.similarity_threshold:
                results.append({
                    "policy_name":  pid,
                    "summary":      meta["summary"],
                    "related_kpis": meta["related_kpis"].split("|"),
                    "similarity":   similarity,
                })
        return results

    def format_context(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return ""
        lines = ["[POLICY KNOWLEDGE CONTEXT]"]
        for r in results:
            lines.append(
                f"POLICY: {r['policy_name']} (similarity: {r['similarity']:.2f})\n"
                f"Related KPIs: {', '.join(r['related_kpis'])}\n"
                f"{r['summary']}\n"
            )
        return "\n".join(lines)

    def query_and_format(self, question: str, n_results: int | None = None) -> str:
        return self.format_context(self.query(question, n_results=n_results))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the OPD policy ChromaDB vector store."
    )
    parser.add_argument(
        "--json", type=Path, default=DEFAULT_JSON_PATH,
        help=f"Path to policy_summaries.json  (default: {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_DIR,
        help=f"Directory to store ChromaDB data  (default: {DEFAULT_DB_DIR})",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name  (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if the DB is already up-to-date.",
    )
    parser.add_argument(
        "--chroma-host", type=str, default=os.getenv("CHROMA_HOST"),
        help="ChromaDB server host (e.g., 'localhost' or 'chroma-server'). If not provided, uses local persistent client.",
    )
    parser.add_argument(
        "--embedding-url", type=str, default=os.getenv("EMBEDDING_SERVER_URL"),
        help="Remote embedding server URL.",
    )
    args = parser.parse_args()

    build_db(
        json_path=args.json,
        db_dir=args.db,
        model_name=args.model,
        force=args.force,
        chroma_host=args.chroma_host,
        embedding_server_url=args.embedding_url,
    )
