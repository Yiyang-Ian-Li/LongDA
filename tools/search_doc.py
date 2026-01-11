import re
from typing import List, Optional

from smolagents import Tool

from .common import format_doc_header, load_document_units


class SearchDocTool(Tool):
    name = "search_doc"
    description = (
        "Searches for a keyword inside a document (PDF, TXT, JSON, PY, CSV, XLS, XLSX). "
        "Supports optional control over the number of matches and surrounding context length."
    )
    inputs = {
        "doc_path": {
            "type": "string",
            "description": "Path to the document to search.",
        },
        "keyword": {
            "type": "string",
            "description": "Keyword to search for.",
        },
        "max_matches": {
            "type": "integer",
            "description": "Maximum number of matches to return. Optional.",
            "nullable": True,
        },
        "context_chars": {
            "type": "integer",
            "description": "Characters of context to include before and after each match. Optional.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, default_max_matches: Optional[int] = None, default_context_chars: int = 150):
        super().__init__()
        self.default_max_matches = default_max_matches
        self.default_context_chars = default_context_chars

    def forward(
        self,
        doc_path: str,
        keyword: str,
        max_matches: Optional[int] = None,
        context_chars: Optional[int] = None,
    ) -> str:
        keyword = (keyword or "").strip()
        if not keyword:
            return "Error: keyword must be a non-empty string."

        try:
            units = load_document_units(doc_path)
        except FileNotFoundError:
            return f"{format_doc_header(doc_path)} Document not found."
        except ValueError as exc:
            return f"{format_doc_header(doc_path)} {exc}"

        context = context_chars if context_chars is not None else self.default_context_chars
        context = max(0, context if context is not None else 0)

        limit = max_matches if max_matches is not None else self.default_max_matches
        limit = None if (limit is None or limit <= 0) else limit

        matches: List[str] = []
        lowered_keyword = keyword.lower()
        pattern = re.escape(lowered_keyword)

        for unit in units:
            text = unit["text"]
            lowered_text = text.lower()
            for occurrence in re.finditer(pattern, lowered_text):
                start, end = occurrence.span()
                snippet_start = max(0, start - context)
                snippet_end = min(len(text), end + context)
                snippet = text[snippet_start:snippet_end]
                matches.append(f"{unit['label']}: ...{snippet}...")
                if limit and len(matches) >= limit:
                    break
            if limit and len(matches) >= limit:
                break

        header = format_doc_header(doc_path)
        if not matches:
            return f"{header} No matches for '{keyword}'."

        body = "\n".join(matches)
        return (
            f"{header} Found {len(matches)} matches for '{keyword}'.\n"
            f"{body}"
        )
