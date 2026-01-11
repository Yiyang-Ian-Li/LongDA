from typing import Any, Dict, List

from smolagents import Tool


class NotesTool(Tool):
    name = "notes"
    description = (
        "Record any notes in free-form text. "
        "Use action='add' with `content` to add a note, "
        "action='view' with `index` to see a specific note, "
        "action='update' with `index` and `content` to modify a note, "
        "action='list' to review all notes, "
        "and action='clear' to reset notes for the current block."
    )
    inputs = {
        "action": {
            "type": "string",
            "description": "One of 'add', 'view', 'update', 'list', or 'clear'. Defaults to 'add'.",
            "nullable": True,
        },
        "content": {
            "type": "string",
            "description": "Note content in any format (required for 'add' and 'update').",
            "nullable": True,
        },
        "index": {
            "type": "integer",
            "description": "Note index (1-based) for 'view' or 'update' actions.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, notes_store: Dict[str, List[str]], block_id: str):
        super().__init__()
        self.block_id = block_id
        self.notes_store = notes_store
        self.notes_store.setdefault(self.block_id, [])

    def forward(
        self,
        action: str = "add",
        content: str | None = None,
        index: int | None = None,
    ) -> str:
        action_clean = (action or "add").strip().lower()
        entries = self.notes_store[self.block_id]

        if action_clean == "add":
            if not content:
                return "Error: 'content' is required when adding a note."
            entries.append(content.strip())
            return f"Added note #{len(entries)}."

        elif action_clean == "view":
            if index is None:
                return "Error: 'index' is required for view action."
            if index < 1 or index > len(entries):
                return f"Error: Invalid index. Valid range: 1-{len(entries)}"
            return f"Note #{index}:\n{entries[index - 1]}"

        elif action_clean == "update":
            if index is None:
                return "Error: 'index' is required for update action."
            if not content:
                return "Error: 'content' is required for update action."
            if index < 1 or index > len(entries):
                return f"Error: Invalid index. Valid range: 1-{len(entries)}"
            entries[index - 1] = content.strip()
            return f"Updated note #{index}."

        elif action_clean == "list":
            if not entries:
                return "No notes recorded yet for this block."
            lines = [
                f"{idx + 1}. {note[:100]}{'...' if len(note) > 100 else ''}"
                for idx, note in enumerate(entries)
            ]
            return "Notes:\n" + "\n".join(lines)

        elif action_clean == "clear":
            entries.clear()
            return "Cleared all notes for the current block."

        else:
            return "Error: action must be 'add', 'view', 'update', 'list', or 'clear'."
