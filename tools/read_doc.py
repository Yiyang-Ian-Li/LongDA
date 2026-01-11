from typing import List

from smolagents import Tool

from .common import (
    format_doc_header,
    load_document_units,
    parse_index_selection,
    summarize_units,
)


class ReadDocTool(Tool):
    name = "read_doc"
    description = (
        "Reads specific portions of a document. Supports PDF, TXT, JSON, PY, CSV, XLS, and XLSX files. "
        "If no range is provided, returns the available range and a short preview."
    )
    inputs = {
        "doc_path": {
            "type": "string",
            "description": "Path to the document to read.",
        },
        "range_str": {
            "type": "string",
            "description": "Optional range like '1-3' or '2,5,7'. Uses one-based numbering.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, doc_path: str, range_str: str = "") -> str:
        try:
            units = load_document_units(doc_path)
        except FileNotFoundError:
            return f"{format_doc_header(doc_path)} Document not found."
        except ValueError as exc:
            return f"{format_doc_header(doc_path)} {exc}"

        total = len(units)
        if total == 0:
            return f"{format_doc_header(doc_path)} Document is empty."

        header = format_doc_header(doc_path)
        kind = units[0]["kind"]
        plural = f"{kind}s"

        if not range_str:
            preview_count = min(5, total)
            preview = summarize_units(units[:preview_count])
            return (
                f"{header} {plural} available: 1-{total}\n"
                f"Sample:\n{preview}"
            )

        indices = parse_index_selection(range_str, total)
        if not indices:
            return (
                f"{header} Unable to parse range '{range_str}'. "
                f"Valid {plural.lower()} span 1-{total}."
            )

        selected_units: List[dict] = [units[i] for i in indices]
        labels = ", ".join(unit["label"] for unit in selected_units)
        body = summarize_units(selected_units)
        return f"{header} {plural}: {labels}\n{body}"
