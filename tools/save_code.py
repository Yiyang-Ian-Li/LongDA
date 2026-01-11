from typing import Dict

from smolagents import Tool


class SaveCodeTool(Tool):
    name = "save_code"
    description = (
        "Stores the Python code used for the current survey/source block. "
        "Call this after recording answers for the block."
    )
    inputs = {
        "code": {
            "type": "string",
            "description": "Python code snippet used to compute the block's answers.",
        },
    }
    output_type = "string"

    def __init__(self, code_store: Dict[str, str], block_id: str):
        super().__init__()
        self.code_store = code_store
        self.block_id = block_id

    def forward(self, code: str) -> str:
        code = (code or "").strip()
        if not code:
            return "Error: code cannot be empty."

        self.code_store[self.block_id] = code
        return f"Code stored for block {self.block_id}."
