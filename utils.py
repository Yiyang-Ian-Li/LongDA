import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _standardize_numeric_text(text: str) -> str:
    """
    Normalize numeric annotations (%, ±, thousand separators) for safer parsing.
    """
    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "±": "+-",
        "+/-": "+-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Drop ± tolerances, keep the central value.
    text = re.sub(
        r"([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)\s*\+-\s*[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?",
        r"\1",
        text,
    )
    text = text.replace("%", "")
    # Remove thousand separators that sit between digits (e.g., 1,234).
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return text


def _try_literal_eval(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def _extract_numbers(text: str) -> Iterable[float]:
    return [float(match) for match in _NUMBER_PATTERN.findall(text)]


def _extract_key_value_pairs(text: str) -> Dict[str, Any]:
    """
    Attempt to extract simple "key: value" pairs from free-form text.
    """
    pattern = re.compile(
        r"([A-Za-z0-9_\-/\s]+?)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )
    pairs = pattern.findall(text)
    if not pairs:
        return {}
    return {key.strip(): float(value) for key, value in pairs}


def _collect_files(path: Path) -> List[str]:
    if not path.exists():
        return []

    collected: List[str] = []
    entries = sorted(path.iterdir(), key=lambda item: item.name.lower())
    for entry in entries:
        if entry.is_file():
            collected.append(str(entry))
        elif entry.is_dir():
            collected.extend(_collect_files(entry))
    return collected


def get_data_and_doc_paths(survey):
    data_files = []
    doc_files = []
    base = Path(f"benchmark/{survey}")
    data_dir = base / "data"
    docs_dir = base / "docs"
    data_files.extend(_collect_files(data_dir))
    doc_files.extend(_collect_files(docs_dir))

    # Deduplicate while preserving order.
    def _unique(sequence):
        seen = set()
        for item in sequence:
            if item not in seen:
                seen.add(item)
                yield item

    return list(_unique(data_files)), list(_unique(doc_files))


def normalize_answer_value(value: Any) -> Any:
    """
    Convert agent or benchmark answers into comparable numeric structures.
    Returns floats, lists, dicts, or None when parsing fails.
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, dict):
        return {
            str(key): normalize_answer_value(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        normalized = [normalize_answer_value(item) for item in value]
        return normalized

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None

        cleaned = _standardize_numeric_text(stripped)
        literal = _try_literal_eval(cleaned)
        if literal is not None:
            return normalize_answer_value(literal)

        kv_pairs = _extract_key_value_pairs(cleaned)
        if kv_pairs:
            return {key: normalize_answer_value(val) for key, val in kv_pairs.items()}

        numbers = _extract_numbers(cleaned)
        if numbers:
            if len(numbers) == 1:
                return float(numbers[0])
            return [float(num) for num in numbers]

        return None

    # Fallback: attempt to serialize complex objects, otherwise return None.
    try:
        return normalize_answer_value(json.loads(value))
    except Exception:
        return None


def build_answer_entry(raw_answer: Any, code: str = "") -> Dict[str, Any]:
    """
    Prepare the payload written to disk for each question.
    """
    raw_text = raw_answer if isinstance(raw_answer, str) else json.dumps(raw_answer, ensure_ascii=False)
    entry = {
        "raw_answer": raw_text,
        "parsed_answer": normalize_answer_value(raw_answer),
        "code": code.strip() if isinstance(code, str) else "",
    }
    return entry
