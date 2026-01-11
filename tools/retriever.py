from typing import Dict, List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from smolagents import Tool

from .common import format_doc_header, load_document_units


class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Retrieves relevant segments from a document (PDF, TXT, JSON, PY, CSV, XLS, XLSX) given a search query. "
        "Supports optional control over the number of matches and surrounding context length."
    )
    inputs = {
        "doc_path": {
            "type": "string",
            "description": "Path to the document to search.",
        },
        "query": {
            "type": "string",
            "description": "Search query in declarative form.",
        },
        "max_matches": {
            "type": "integer",
            "description": "Maximum number of segments to return. Optional.",
            "nullable": True,
        },
        "context_chars": {
            "type": "integer",
            "description": "Characters of context to include around the best match within each segment. Optional.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        default_max_matches: Optional[int] = None,
        default_context_chars: int = 200,
    ):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        self.default_max_matches = default_max_matches
        self.default_context_chars = default_context_chars
        self._cache: Dict[str, Dict] = {}

    def _ensure_retriever(self, doc_path: str) -> Dict:
        if doc_path in self._cache:
            return self._cache[doc_path]

        units = load_document_units(doc_path)
        documents = []
        chunk_counter = 0
        for unit in units:
            text = unit["text"]
            if not text:
                continue
            metadata = {
                "label": unit["label"],
                "kind": unit["kind"],
                "source": doc_path,
            }
            chunks = self.text_splitter.create_documents([text], metadatas=[metadata])
            for chunk in chunks:
                chunk.metadata["chunk_index"] = chunk_counter
                chunk_counter += 1
                documents.append(chunk)

        if not documents:
            raise ValueError("Document has no readable content.")

        retriever = BM25Retriever.from_documents(documents)
        self._cache[doc_path] = {"retriever": retriever}
        return self._cache[doc_path]

    def _format_snippet(self, content: str, query: str, context_chars: Optional[int]) -> str:
        query = query.strip()
        if not content:
            return ""
        if context_chars is None or context_chars <= 0:
            return content

        lowered = content.lower()
        position = lowered.find(query.lower())
        if position == -1:
            return content[:context_chars]

        start = max(0, position - context_chars)
        end = min(len(content), position + len(query) + context_chars)
        return content[start:end]

    def forward(
        self,
        doc_path: str,
        query: str,
        max_matches: Optional[int] = None,
        context_chars: Optional[int] = None,
    ) -> str:
        query = (query or "").strip()
        if not query:
            return "Error: query must be a non-empty string."

        try:
            cached = self._ensure_retriever(doc_path)
        except FileNotFoundError:
            return f"{format_doc_header(doc_path)} Document not found."
        except ValueError as exc:
            return f"{format_doc_header(doc_path)} {exc}"

        retriever: BM25Retriever = cached["retriever"]

        request_limit = max_matches if max_matches is not None else self.default_max_matches
        request_limit = None if (request_limit is None or request_limit <= 0) else request_limit

        if request_limit:
            retriever.k = request_limit

        documents = retriever.invoke(query)
        if request_limit:
            documents = documents[:request_limit]

        if not documents:
            return f"{format_doc_header(doc_path)} No matches for '{query}'."

        context_window = (
            context_chars
            if context_chars is not None
            else self.default_context_chars
        )

        lines: List[str] = []
        for doc in documents:
            metadata = doc.metadata or {}
            label = metadata.get("label", "Segment")
            chunk_id = metadata.get("chunk_index")
            snippet = self._format_snippet(doc.page_content, query, context_window)
            if chunk_id is not None:
                lines.append(f"{label} (chunk {chunk_id}): ...{snippet}...")
            else:
                lines.append(f"{label}: ...{snippet}...")

        header = format_doc_header(doc_path)
        return (
            f"{header} Top {len(lines)} segments for '{query}'.\n"
            + "\n".join(lines)
        )
