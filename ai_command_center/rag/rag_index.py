"""
RAG index — build and query a local vector-store over documents in a folder.

Supports:
  - PDF, DOCX, TXT, MD, HTML, CSV files
  - ChromaDB as the vector store (persisted on disk)
  - sentence-transformers for dense embeddings
  - Automatic incremental re-indexing (only indexes changed/new files)

Google Drive sync path (Windows):
    C:\\Users\\Administrator\\Documents

Configure via environment variables RAG_DOCS_DIR and RAG_INDEX_DIR.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from ai_command_center import config

logger = logging.getLogger(__name__)

# ── Supported file extensions ───────────────────────────────────────────────
_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm", ".csv"}


def _read_file(path: Path) -> str:
    """Extract plain text from a file based on its extension."""
    ext = path.suffix.lower()

    if ext in {".txt", ".md"}:
        return path.read_text(errors="replace")

    if ext == ".pdf":
        try:
            import pymupdf  # PyMuPDF — faster and more accurate than pypdf2
            doc = pymupdf.open(str(path))
            return "\n".join(page.get_text() for page in doc)
        except ImportError:
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                return "\n".join(p.extract_text() or "" for p in reader.pages)
            except ImportError:
                logger.warning("No PDF library found; skipping %s. Install pymupdf.", path)
                return ""

    if ext == ".docx":
        try:
            from docx import Document
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            logger.warning("python-docx not installed; skipping %s.", path)
            return ""

    if ext in {".html", ".htm"}:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(path.read_bytes(), "html.parser")
            return soup.get_text(separator="\n")
        except ImportError:
            return path.read_text(errors="replace")

    if ext == ".csv":
        return path.read_text(errors="replace")

    return ""


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


class RAGIndex:
    """Persistent RAG index over a local document directory."""

    _META_FILE = "rag_meta.json"

    def __init__(
        self,
        docs_dir: str  = config.RAG_DOCS_DIR,
        index_dir: str = config.RAG_INDEX_DIR,
        embedding_model: str = config.RAG_EMBEDDING_MODEL,
        chunk_size: int      = config.RAG_CHUNK_SIZE,
        chunk_overlap: int   = config.RAG_CHUNK_OVERLAP,
    ) -> None:
        self.docs_dir      = Path(docs_dir)
        self.index_dir     = Path(index_dir)
        self.embedding_model = embedding_model
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.index_dir / self._META_FILE

        self._collection = None   # ChromaDB collection (lazy)
        self._embedder   = None   # SentenceTransformer (lazy)

    # ── Lazy loading ────────────────────────────────────────────────────────
    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
            except ImportError as exc:
                raise ImportError(
                    "chromadb is required. Run: pip install chromadb"
                ) from exc

            client = chromadb.PersistentClient(path=str(self.index_dir))
            self._collection = client.get_or_create_collection(
                name     = "rag_documents",
                metadata = {"hnsw:space": "cosine"},
            )
        return self._collection

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Run: pip install sentence-transformers"
                ) from exc

            logger.info("Loading embedding model: %s …", self.embedding_model)
            self._embedder = SentenceTransformer(self.embedding_model)
            logger.info("Embedding model loaded.")
        return self._embedder

    # ── Metadata helpers ────────────────────────────────────────────────────
    def _load_meta(self) -> dict:
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                return json.load(f)
        return {}

    def _save_meta(self, meta: dict) -> None:
        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ── Public API ──────────────────────────────────────────────────────────
    def build(self, force: bool = False) -> dict[str, int]:
        """Scan docs_dir and (re-)index any new or changed files.

        Args:
            force: If True, re-index ALL files regardless of change status.

        Returns:
            dict with ``indexed``, ``skipped``, ``errors`` counts.
        """
        meta = {} if force else self._load_meta()
        collection = self._get_collection()
        embedder   = self._get_embedder()

        stats = {"indexed": 0, "skipped": 0, "errors": 0}

        if not self.docs_dir.exists():
            logger.warning("Documents directory does not exist: %s", self.docs_dir)
            return stats

        all_files = [
            p for p in self.docs_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        ]
        logger.info("Found %d supported files in %s", len(all_files), self.docs_dir)

        for fpath in all_files:
            try:
                fhash = _file_hash(fpath)
                str_path = str(fpath)

                if not force and meta.get(str_path) == fhash:
                    stats["skipped"] += 1
                    continue

                text = _read_file(fpath)
                if not text.strip():
                    stats["skipped"] += 1
                    continue

                chunks = _chunk_text(text, self.chunk_size, self.chunk_overlap)
                embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

                ids       = [f"{str_path}::{i}" for i in range(len(chunks))]
                metadatas = [{"source": str_path, "chunk": i} for i in range(len(chunks))]

                # Delete old chunks for this file before adding new ones
                try:
                    existing = collection.get(where={"source": str_path})
                    if existing["ids"]:
                        collection.delete(ids=existing["ids"])
                except Exception:
                    pass

                collection.add(
                    ids        = ids,
                    embeddings = embeddings,
                    documents  = chunks,
                    metadatas  = metadatas,
                )

                meta[str_path] = fhash
                stats["indexed"] += 1
                logger.debug("Indexed: %s (%d chunks)", fpath.name, len(chunks))

            except Exception as exc:
                logger.error("Error indexing %s: %s", fpath, exc, exc_info=True)
                stats["errors"] += 1

        self._save_meta(meta)
        logger.info(
            "Index build complete — indexed: %d, skipped: %d, errors: %d",
            stats["indexed"], stats["skipped"], stats["errors"],
        )
        return stats

    def query(self, query: str, top_k: int = config.RAG_TOP_K) -> list[dict[str, Any]]:
        """Semantic search over the index.

        Args:
            query: The search query.
            top_k: Number of chunks to return.

        Returns:
            List of dicts with keys ``text``, ``source``, ``score``.
        """
        collection = self._get_collection()
        embedder   = self._get_embedder()

        q_embedding = embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings = q_embedding,
            n_results        = top_k,
            include          = ["documents", "metadatas", "distances"],
        )

        docs = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            docs.append({
                "text":   doc,
                "source": meta.get("source", ""),
                "score":  round(1 - dist, 4),  # cosine distance → similarity
            })

        return docs

    def stats(self) -> dict[str, int]:
        """Return basic statistics about the current index."""
        collection = self._get_collection()
        meta       = self._load_meta()
        return {
            "total_chunks":     collection.count(),
            "indexed_files":    len(meta),
            "docs_dir":         str(self.docs_dir),
            "index_dir":        str(self.index_dir),
        }
