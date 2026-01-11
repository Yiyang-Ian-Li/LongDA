import re
from smolagents import Tool


class PromptTool(Tool):
    name = "prompt"
    description = (
        "Displays the current benchmark prompt. "
        "Use without q_id to see the full prompt, "
        "or provide a q_id (e.g., 'ATUS_pubs_q1') to see details for a specific question."
    )
    inputs = {
        "q_id": {
            "type": "string",
            "description": "Optional question ID to view specific question details.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, prompt: str | None = None):
        super().__init__()
        self.prompt = prompt

    def forward(self, q_id: str | None = None) -> str:
        if not self.prompt:
            return "No prompt available"
        
        if not q_id:
            return self.prompt
        
        # Extract specific question details
        pattern = rf"Question ID: {re.escape(q_id)}\n(.*?)(?=\nQuestion ID:|\nAvailable data files:|$)"
        match = re.search(pattern, self.prompt, re.DOTALL)
        
        if not match:
            return f"Question {q_id} not found in prompt. Use check_prompt() without q_id to see all questions."
        
        return f"Question ID: {q_id}\n{match.group(1).strip()}"
