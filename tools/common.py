import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import PyPDF2


WHITESPACE_PATTERN = re.compile(r"\s+")
TEXT_EXTENSIONS = {".txt", ".json", ".py"}
TABULAR_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def clean_text(text: str) -> str:
    """Collapse whitespace to keep snippets compact."""
    return WHITESPACE_PATTERN.sub(" ", text or "").strip()


def parse_index_selection(range_str: str, total: int) -> List[int]:
    """Parse '1-3,5' into zero-based indices within [0, total)."""
    if not range_str:
        return []

    indices: List[int] = []
    for token in range_str.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            try:
                start = max(0, int(start_text) - 1)
                end = int(end_text)
            except ValueError:
                continue
            for value in range(start, min(end, total)):
                if value not in indices:
                    indices.append(value)
        else:
            try:
                idx = int(token) - 1
            except ValueError:
                continue
            if 0 <= idx < total and idx not in indices:
                indices.append(idx)
    return indices


def _load_pdf_units(path: Path) -> List[Dict]:
    units: List[Dict] = []
    with path.open("rb") as handle:
        reader = PyPDF2.PdfReader(handle)
        for idx, page in enumerate(reader.pages):
            text = clean_text(page.extract_text() or "")
            units.append(
                {
                    "kind": "Page",
                    "label": f"Page {idx + 1}",
                    "index": idx,
                    "text": text,
                }
            )
    return units


def _load_text_units(path: Path) -> List[Dict]:
    units: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            units.append(
                {
                    "kind": "Line",
                    "label": f"Line {idx + 1}",
                    "index": idx,
                    "text": clean_text(line),
                }
            )
    return units


def _load_json_units(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
            formatted = json.dumps(payload, indent=2, ensure_ascii=False)
            text_lines = formatted.splitlines()
        except json.JSONDecodeError:
            handle.seek(0)
            text_lines = handle.readlines()

    units: List[Dict] = []
    for idx, line in enumerate(text_lines):
        units.append(
            {
                "kind": "Line",
                "label": f"Line {idx + 1}",
                "index": idx,
                "text": clean_text(line),
            }
        )
    return units


def _load_tabular_units(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_excel(path)

    columns = list(frame.columns)
    units: List[Dict] = []
    for idx, (_, row) in enumerate(frame.iterrows()):
        values = " | ".join(f"{col}: {row[col]}" for col in columns)
        units.append(
            {
                "kind": "Row",
                "label": f"Row {idx + 1}",
                "index": idx,
                "text": clean_text(str(values)),
            }
        )
    return units


def load_document_units(doc_path: str) -> List[Dict]:
    """Return a list of units (pages/lines/rows) for supported document types."""
    path = Path(doc_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_units(path)
    if suffix == ".json":
        return _load_json_units(path)
    if suffix in TEXT_EXTENSIONS:
        return _load_text_units(path)
    if suffix in TABULAR_EXTENSIONS:
        return _load_tabular_units(path)

    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


def summarize_units(units: Iterable[Dict], join_text: bool = True) -> str:
    """Format units for display."""
    lines: List[str] = []
    for unit in units:
        content = unit.get("text", "")
        if join_text:
            lines.append(f"{unit.get('label')}: {content}")
        else:
            lines.append(f"{unit.get('label')}")
    return "\n".join(lines).strip()


def format_doc_header(doc_path: str) -> str:
    # path = Path(doc_path)
    # suffix = path.suffix[1:].upper() if path.suffix else "FILE"
    # return f"[{suffix}:{path.name}]"
    return f"[{doc_path}]"
