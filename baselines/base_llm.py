import json
import os
from pathlib import Path
from datetime import datetime, timezone
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Sequence

from smolagents import LiteLLMModel


class BaseLLM:
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model = self._initialize_model(config.get("model", {}))

        self.recorded_answers: Dict[str, Dict[str, str]] = {}
        self.code_snippets: Dict[str, str] = {}
        self.notes_store: Dict[str, List[Dict[str, str]]] = {}

    def _initialize_model(self, model_config: Dict[str, Any]) -> Any:
        """
        Initialize the model based on the provided configuration.
        """
        model = LiteLLMModel(
            model_id=model_config.get("id"),
            api_key=model_config.get("api_key"),
            api_base=model_config.get("api_base"),
        )
        return model

    def run_block(self, block: Dict[str, Any]) -> None:
        """
        Execute the agent for a single benchmark slice described by the supplied block dict.
        The block must include survey metadata, questions, data paths, and context.
        """
        
        return
    
    def collect_artifacts(self) -> Dict[str, Any]:
        """
        Collect artifacts generated during the run, such as recorded answers, code snippets, notes, and tool usage counts.
        """
        return {
            "recorded_answers": self.recorded_answers,
            "code_snippets": self.code_snippets,
            "notes_store": self.notes_store,
        }