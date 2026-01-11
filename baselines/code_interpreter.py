import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from my_agent import build_prompt


def _flatten(iterable: Iterable[Iterable[str]]) -> List[str]:
    items: List[str] = []
    for group in iterable:
        items.extend(group)
    return items


class CodeInterpreterBaseline:
    """
    Runner that delegates each benchmark block to OpenAI's Code Interpreter (Responses API).

    The runner uploads the relevant survey files to OpenAI's file container, crafts a block
    prompt identical to the agent baseline, and asks the model to return JSON-formatted answers.
    Collected outputs are normalised so the existing evaluation pipeline can re-use them.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        model_cfg = config.get("model", {})

        model_id = os.getenv("CODE_INTERPRETER_MODEL_ID", model_cfg.get("id", "gpt-4.1"))
        api_key = os.getenv("CODE_INTERPRETER_API_KEY", model_cfg.get("api_key"))
        api_base = os.getenv("CODE_INTERPRETER_API_BASE", model_cfg.get("api_base"))
        organization = os.getenv("CODE_INTERPRETER_ORG", model_cfg.get("organization"))

        client_kwargs: Dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if api_base:
            client_kwargs["base_url"] = api_base
        if organization:
            client_kwargs["organization"] = organization

        self.client = OpenAI(**client_kwargs)
        self.model_id = model_id

        self.recorded_answers: Dict[str, Dict[str, str]] = {}
        self.code_snippets: Dict[str, str] = {}
        self.notes_store: Dict[str, List[Dict[str, str]]] = {}
        self.tool_usage_counts: Dict[str, int] = {"responses_calls": 0}
        self._file_cache: Dict[str, str] = {}

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _upload_file(self, path: str) -> Optional[str]:
        if path in self._file_cache:
            return self._file_cache[path]

        file_path = Path(path)
        if not file_path.exists():
            print(f"Warning: Skipping missing file {file_path}")
            return None

        try:
            with file_path.open("rb") as handle:
                uploaded = self.client.files.create(file=handle, purpose="assistants")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: Failed to upload {file_path}: {exc}")
            return None

        self._file_cache[path] = uploaded.id
        return uploaded.id

    def _ensure_attachments(self, file_paths: Iterable[str]) -> List[Dict[str, Any]]:
        attachments: List[Dict[str, Any]] = []
        for path in file_paths:
            file_id = self._upload_file(path)
            if file_id:
                attachments.append(
                    {
                        "file_id": file_id,
                        "tools": [{"type": "code_interpreter"}],
                    }
                )
        return attachments

    def _collect_text_output(self, response: Any) -> str:
        """
        Normalise the variegated shape of the Responses API outputs into a single string.
        """
        text = getattr(response, "output_text", None)
        if text:
            return text

        dump_method = getattr(response, "model_dump", None)
        data = dump_method() if callable(dump_method) else getattr(response, "to_dict", lambda: {})()
        output_items = data.get("output", [])

        chunks: List[str] = []
        for item in output_items:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                content_type = content.get("type")
                if content_type == "output_text":
                    chunks.append(content.get("text") or "")
                elif content_type == "text":
                    text_obj = content.get("text")
                    if isinstance(text_obj, dict):
                        chunks.append(text_obj.get("value", ""))
                    elif isinstance(text_obj, str):
                        chunks.append(text_obj)
        return "\n".join(chunk for chunk in chunks if chunk).strip()

    def _extract_json_payload(self, text: str) -> Optional[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Attempt to recover the first JSON object inside the response.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None
        return None

    def _parse_answers(
        self,
        payload: Dict[str, Any],
        question_ids: List[str],
        block_id: str,
    ) -> None:
        answers = payload.get("answers")
        if isinstance(answers, dict):
            for q_id in question_ids:
                value = answers.get(q_id)
                if value is None:
                    continue
                self.recorded_answers[q_id] = {"raw_answer": str(value), "code": ""}

        # Optional code snippet at block level.
        block_code = payload.get("analysis_code") or payload.get("code")
        if isinstance(block_code, str):
            self.code_snippets[block_id] = block_code.strip()

        # Optional note capture for parity with agent workflow.
        block_notes = payload.get("column_notes")
        if isinstance(block_notes, list):
            valid_notes: List[Dict[str, str]] = []
            for entry in block_notes:
                if not isinstance(entry, dict):
                    continue
                column = str(entry.get("column", "")).strip()
                meaning = str(entry.get("meaning", "")).strip()
                if column and meaning:
                    valid_notes.append({"column": column, "meaning": meaning})
            if valid_notes:
                self.notes_store[block_id] = valid_notes

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run_block(self, block: Dict[str, Any]) -> None:
        prompt = build_prompt(
            instructions=self.config.get("instructions", ""),
            survey=block["survey"],
            source=block["source"],
            questions=block["questions"],
            answer_structures=block["answer_structures"],
            additional_infos=block["additional_infos"],
            data_paths=block["data_paths"],
            doc_paths=block["doc_paths"],
        )

        question_ids = [
            f"{block['survey']}_{block['source']}_q{i}"
            for i in range(1, len(block["questions"]) + 1)
        ]

        file_paths = _flatten([block["data_paths"], block["doc_paths"]])
        attachments = self._ensure_attachments(file_paths)

        if not attachments:
            print(f"Warning: No files uploaded for block {block['block_id']}.")

        system_instructions = self.config.get(
            "system_prompt",
            (
                "You are an analytical data scientist using Python code execution. "
                "Read the provided files and answer each question precisely. "
                "Return ONLY a JSON object matching this schema:\n"
                "{\n"
                '  "answers": {\n'
                '    "survey_source_q1": "final answer string",\n'
                '    "...": "..."\n'
                "  },\n"
                '  "analysis_code": "optional consolidated python code you executed",\n'
                '  "column_notes": [ {"column": "...", "meaning": "..."}, ... ]\n'
                "}\n"
                "Ensure numeric answers respect the requested formats."
            ),
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_instructions}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]

        print(f"\nRunning OpenAI Code Interpreter for {block['survey']} ({block['source']}) ...")

        try:
            self.tool_usage_counts["responses_calls"] += 1
            # Prefer the convenience polling helper when available.
            responder = getattr(self.client.responses, "create_and_poll", None)
            if callable(responder):
                response = responder(
                    model=self.model_id,
                    input=messages,
                    attachments=attachments or None,
                )
            else:
                response = self.client.responses.create(
                    model=self.model_id,
                    input=messages,
                    attachments=attachments or None,
                )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error: Failed to invoke Code Interpreter for block {block['block_id']}: {exc}")
            return

        text_output = self._collect_text_output(response)
        payload = self._extract_json_payload(text_output)
        if payload is None:
            print(
                f"Warning: Unable to parse JSON output for block {block['block_id']}. "
                "Raw response will be stored as-is."
            )
            for q_id in question_ids:
                if q_id not in self.recorded_answers:
                    self.recorded_answers[q_id] = {"raw_answer": text_output, "code": ""}
            return

        self._parse_answers(payload, question_ids, block["block_id"])

    def collect_artifacts(self) -> Dict[str, Any]:
        uploaded_count = len(self._file_cache)
        if uploaded_count:
            self.tool_usage_counts["uploaded_files"] = uploaded_count

        return {
            "recorded_answers": self.recorded_answers,
            "code_snippets": self.code_snippets,
            "notes_store": self.notes_store,
            "tool_usage_counts": self.tool_usage_counts,
        }
