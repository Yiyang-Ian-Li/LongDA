import math
import re
from typing import Dict

from smolagents import Tool


class AnswerTool(Tool):
    name = "answer"
    description = (
        "Manage answers for questions. "
        "Use action='add' or 'update' to store/modify an answer, "
        "action='view' to see a specific answer, "
        "or action='list' to see all recorded answers. "
        "IMPORTANT: Answer must be a number or list of numbers (e.g., 45.3 or [12.5, 34.7]). "
        "Call `save_code` separately if you need to record the code used."
    )
    inputs = {
        "action": {
            "type": "string",
            "description": "One of 'add', 'update', 'view', or 'list'. Defaults to 'add'.",
            "nullable": True,
        },
        "q_id": {
            "type": "string",
            "description": "Question ID in the format 'survey_source_qnum'. Required for add/update/view.",
            "nullable": True,
        },
        "answer": {
            "type": "any",
            "description": "The final answer as a number or list of numbers. Can be: number (45.3), list ([12.5, 34.7]), or string ('45.3'). Required for add/update.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, answers_dict: Dict[str, Dict] | None = None, block_id: str = ""):
        super().__init__()
        self.answers = answers_dict if answers_dict is not None else {}
        self.block_id = block_id
    
    def _validate_answer(self, answer) -> tuple[bool, str]:
        """Validate that answer is a number or list of numbers."""
        # If already a number or list, validate directly
        if isinstance(answer, (int, float)):
            return True, ""
        if isinstance(answer, list):
            if all(isinstance(x, (int, float)) for x in answer):
                return True, ""
            return False, "List must contain only numbers"
        
        # If string, try to parse it
        if isinstance(answer, str):
            import ast
            try:
                parsed = ast.literal_eval(answer)
                if isinstance(parsed, (int, float)):
                    return True, ""
                if isinstance(parsed, list):
                    if all(isinstance(x, (int, float)) for x in parsed):
                        return True, ""
                    return False, "List must contain only numbers"
                return False, "Answer must be a number or list of numbers"
            except (ValueError, SyntaxError):
                return False, f"Invalid format. Use number (45.3) or list ([12.5, 34.7])"
        
        return False, "Answer must be a number, list of numbers, or string representation"

    def forward(self, action: str = "add", q_id: str | None = None, answer = None) -> str:
        action_clean = (action or "add").strip().lower()

        if action_clean in ["add", "update"]:
            if not q_id:
                return "Error: q_id is required for add/update action."
            if answer is None:
                return "Error: answer is required for add/update action."
            
            # Validate answer format
            is_valid, error_msg = self._validate_answer(answer)
            if not is_valid:
                return f"Error: {error_msg}. Provide answer as number (45.3) or list ([12.5, 34.7])"

            # Convert answer to string for storage
            if isinstance(answer, str):
                answer_str = answer
            else:
                answer_str = str(answer)
            
            entry = self.answers.setdefault(q_id, {"raw_answer": "", "code": ""})
            entry["raw_answer"] = answer_str
            return f"Answer recorded for question {q_id}"

        elif action_clean == "view":
            if not q_id:
                return "Error: q_id is required for view action."
            if q_id not in self.answers:
                return f"No answer found for question {q_id}"
            entry = self.answers[q_id]
            answer_text = entry.get("raw_answer", "")
            code_text = entry.get("code", "")
            result = f"Question {q_id}:\nAnswer: {answer_text}"
            if code_text:
                result += f"\nCode: {code_text}"
            return result

        elif action_clean == "list":
            if not self.answers:
                return "No answers recorded yet."
            
            # Filter to current block if block_id is set
            relevant_answers = {
                qid: ans for qid, ans in self.answers.items()
                if not self.block_id or qid.startswith(self.block_id)
            }
            
            if not relevant_answers:
                return f"No answers recorded for block {self.block_id}" if self.block_id else "No answers recorded yet."
            
            lines = []
            for qid in sorted(relevant_answers.keys()):
                ans_text = relevant_answers[qid].get("raw_answer", "")
                preview = ans_text[:100] + "..." if len(ans_text) > 100 else ans_text
                lines.append(f"- {qid}: {preview}")
            return "Recorded Answers:\n" + "\n".join(lines)

        else:
            return "Error: action must be 'add', 'update', 'view', or 'list'."
