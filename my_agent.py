import os
from types import MethodType
from typing import Any, Dict, Iterable, List, Sequence

from smolagents import CodeAgent, OpenAIModel, Tool, LiteLLMModel, REMOVE_PARAMETER
from langfuse import get_client, propagate_attributes
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
        
from tools import (
    PromptTool,
    NotesTool,
    ReadDocTool,
    RetrieverTool,
    SaveCodeTool,
    SearchDocTool,
    AnswerTool,
)

DEFAULT_MAX_STEPS = 50


def build_prompt(
    instructions: str,
    survey: str,
    source: str,
    questions: List[str],
    answer_structures: List[str],
    additional_infos: List[str],
    data_paths: List[str],
    doc_paths: List[str],
) -> str:
    qa_pairs = []
    for i, (question, answer_structure, additional_info) in enumerate(zip(questions, answer_structures, additional_infos), start=1):
        q_id = f"{survey}_{source}_q{i}"
        additional_info = str(additional_info) 
        qa_pairs.append(
            f"Question ID: {q_id}\n"
            f"Q{i}: {question}\n"
            f"Expected Answer Structure: {answer_structure}\n"
            f"Additional Info: {additional_info}\n"
        )

    qa_block = "\n".join(qa_pairs)
    data_block = "\n".join(data_paths) if data_paths else "None provided"
    doc_block = "\n".join(doc_paths) if doc_paths else "None provided"

    return f"""Analyze the survey data and relevant documents to answer the following questions.
Survey: {survey}

Questions and Expected Answer Structures:
{qa_block}

Available data files:
{data_block}

Available documentation files:
{doc_block}

Instructions:
{instructions}

"""


def _instrument_tool(tool: Tool, usage_counts: Dict[str, int]) -> Tool:
    original_forward = tool.forward

    def wrapped_forward(self, *args, **kwargs):
        usage_counts[tool.name] = usage_counts.get(tool.name, 0) + 1
        return original_forward(*args, **kwargs)

    tool.forward = MethodType(wrapped_forward, tool)
    return tool


def _build_tools(
    tool_names: Iterable[str],
    prompt: str,
    answers_store: Dict[str, Dict[str, str]],
    code_store: Dict[str, str],
    notes_store: Dict[str, List[str]],
    block_id: str,
    usage_counts: Dict[str, int],
    config: Dict[str, Any],
) -> List[Tool]:
    retriever_defaults = config.get("retriever_defaults", {})
    search_defaults = config.get("search_defaults", {})

    available_factories = {
        "retriever": lambda: RetrieverTool(
            chunk_size=retriever_defaults.get("chunk_size", 500),
            chunk_overlap=retriever_defaults.get("chunk_overlap", 100),
            default_max_matches=retriever_defaults.get("max_matches", 5),
            default_context_chars=retriever_defaults.get("context_chars", 200),
        ),
        "read_doc": lambda: ReadDocTool(),
        "search_doc": lambda: SearchDocTool(
            default_max_matches=search_defaults.get("max_matches", 5),
            default_context_chars=search_defaults.get("context_chars", 150),
        ),
        "save_code": lambda: SaveCodeTool(code_store, block_id),
        "answer": lambda: AnswerTool(answers_store, block_id),
        "prompt": lambda: PromptTool(prompt),
        "notes": lambda: NotesTool(notes_store, block_id),
    }

    selected_tools: List[Tool] = []
    missing_factories = []
    for name in tool_names:
        factory = available_factories.get(name)
        if factory is None:
            missing_factories.append(name)
            continue
        tool_instance = factory()
        selected_tools.append(_instrument_tool(tool_instance, usage_counts))

    if missing_factories:
        print(f"Warning: Unknown tools requested and skipped: {', '.join(missing_factories)}")

    tool_names_lower = {name.lower() for name in tool_names}
    if "answer" not in tool_names_lower:
        raise ValueError("`answer` tool is required to record benchmark answers. Please include it.")

    return selected_tools


class MyAgentBaseline:
    """
    Baseline runner that prepares the agent once and executes it for each benchmark block.
    The traversal of the benchmark and result evaluation are handled externally.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.langfuse_enabled = False
        langfuse_config = config.get("langfuse", {})
        if langfuse_config.get("use", False):
            self._initialize_langfuse(config)
            self.langfuse_enabled = True
        self.model = self._initialize_model(config.get("model", {}))
        self.tool_selection: Sequence[str] = config.get("enabled_tools", ())

        self.recorded_answers: Dict[str, Dict[str, str]] = {}
        self.code_snippets: Dict[str, str] = {}
        self.notes_store: Dict[str, List[str]] = {}
        self.tool_usage_counts: Dict[str, int] = {}
        
    def _initialize_langfuse(self, config: Dict[str, Any]) -> None:
        langfuse_config = config.get("langfuse", {})
        os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_config.get("public_key", "")
        os.environ["LANGFUSE_SECRET_KEY"] = langfuse_config.get("secret_key", "")
        os.environ["LANGFUSE_HOST"] = langfuse_config.get("host", "https://us.cloud.langfuse.com")
        
        self.langfuse_client = get_client()
        # Verify connection
        if self.langfuse_client.auth_check():
            print("Langfuse client is authenticated and ready!")
        else:
            print("Authentication failed. Please check your credentials and host.")
            
        SmolagentsInstrumentor().instrument()

    def _initialize_model(self, model_cfg: Dict[str, Any]):
        model_id = os.getenv("AGENT_MODEL_ID", model_cfg.get("id"))
        api_key = os.getenv("AGENT_MODEL_API_KEY", model_cfg.get("api_key"))
        api_base = os.getenv("AGENT_MODEL_API_BASE", model_cfg.get("api_base", None))
        rpm = model_cfg.get("rpm", None)
        
        if 'gpt' in model_id:
            return LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                # reasoning_effort='high',
                # api_base=api_base,
                # temperature=0.0,
                # stop=REMOVE_PARAMETER,
            )
        
        if 'claude' in model_id:
            return LiteLLMModel(
                model_id=model_id,
                api_key=api_key,
                api_base=api_base,
                # temperature=0.0,
                requests_per_minute=rpm,
            )
            
        if 'gemini' in model_id:
            return OpenAIModel(
                model_id=model_id,
                api_key=api_key,
                api_base=api_base,
                reasoning_effort='high'
            )

        return OpenAIModel(
            model_id=model_id,
            api_key=api_key,
            api_base=api_base,
        )

    def run_block(self, block: Dict[str, Any], run_timestamp: str = "") -> None:
        """
        Execute the agent for a single benchmark slice described by the supplied block dict.
        The block must include survey metadata, questions, data paths, and context.
        Optionally saves block answers and notes to save_dir after completion.
        """
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

        tools = _build_tools(
            self.tool_selection,
            prompt,
            self.recorded_answers,
            self.code_snippets,
            self.notes_store,
            block["block_id"],
            self.tool_usage_counts,
            self.config,
        )

        agent = CodeAgent(
            tools=tools,
            model=self.model,
            additional_authorized_imports=[
                # Python builtins - allow all
                "builtins.*",
                # Data science packages
                "numpy",
                "numpy.*",
                "pandas",
                "pandas.*",
                "scipy",
                "scipy.*",
                "statsmodels",
                "statsmodels.*",
                "sklearn",
                "sklearn.*",
                # Standard library - essential
                "csv",
                "json",
                "math",
                "os",
                "os.path",
                "pathlib",
                "re",
                "sys",
                "datetime",
                "time",
                # Standard library - data structures & utilities
                "collections",
                "collections.*",
                "itertools",
                "functools",
                "operator",
                "typing",
                # Standard library - numeric & statistical
                "statistics",
                "decimal",
                "fractions",
                "random",
            ],
            executor_kwargs={'additional_functions':{'open': open, 'repr': repr}},
            return_full_result=True,
        )

        max_steps = self.config.get("max_steps", DEFAULT_MAX_STEPS)
        print('Max steps:', max_steps)
        print(f"\nRunning benchmark for {block['survey']} ({block['source']})...")
        
        # Set up Langfuse trace attributes if enabled and run agent
        if self.langfuse_enabled:
            model_id = self.config.get("model", {}).get("id", "unknown")
            # Use run_timestamp as batch identifier
            batch_id = run_timestamp if run_timestamp else "unknown"
            with propagate_attributes(
                user_id=model_id,
                session_id=f"{block['survey']}_{block['source']}",
                metadata={
                    "model": model_id,
                    "survey": block["survey"],
                    "source": block["source"],
                    "block_id": block["block_id"],
                    "num_questions": str(len(block["questions"])),
                    "max_steps": str(max_steps),
                    "batch_id": batch_id,
                },
                tags=[model_id, block["survey"], block["source"]],
                version=batch_id,
                trace_name=f"{block['survey']}_{block['source']}"
            ):
                run_result = self._run_agent_with_retry(agent, prompt, max_steps, block)
        else:
            run_result = self._run_agent_with_retry(agent, prompt, max_steps, block)
        
        # Extract block metrics from RunResult
        block_metrics = {
            "state": run_result.state,
            "tokens": {
                "input_tokens": run_result.token_usage.input_tokens if run_result.token_usage else 0,
                "output_tokens": run_result.token_usage.output_tokens if run_result.token_usage else 0,
                "total_tokens": run_result.token_usage.total_tokens if run_result.token_usage else 0,
            },
            "timing": {
                "seconds": round(run_result.timing.duration, 2) if run_result.timing else 0,
                "minutes": round(run_result.timing.duration / 60, 2) if run_result.timing else 0,
            },
            "steps": len(run_result.messages) - 2,
            "num_questions": len(block["questions"]),
            "messages": run_result.messages,
        }
        
        # Save block results and return metrics
        return self._save_block_results(block, block_metrics)
    
    def _run_agent_with_retry(self, agent, prompt, max_steps, block):
        """Run agent with retry logic for API failures. Returns RunResult."""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                result = agent.run(prompt, max_steps=max_steps)
                return result  # Success, return RunResult
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 60  # 60, 120 seconds
                    print(f"\nAPI error on attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s...")
                    print(f"Error: {error_msg[:200]}")
                    time.sleep(wait_time)
                else:
                    print(f"\nFailed after {max_retries} attempts. Error: {error_msg[:200]}")
                    raise

    def _save_block_results(self, block: Dict[str, Any], block_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract block answers and return them with metrics (no individual files saved)."""
        block_id = block["block_id"]
        
        # Extract block answers for CSV update
        block_answers = {
            qid: ans for qid, ans in self.recorded_answers.items()
            if qid.startswith(block_id)
        }
        
        # Add answers_recorded count to metrics
        block_metrics["answers_recorded"] = len(block_answers)
        
        # Return both answers and metrics
        return {
            "block_answers": block_answers,
            "block_metrics": block_metrics
        }
    
    def collect_artifacts(self) -> Dict[str, Any]:
        """
        Return the recorded answers and bookkeeping produced while running the agent.
        """
        return {
            "recorded_answers": self.recorded_answers,
            "code_snippets": self.code_snippets,
            "notes_store": self.notes_store,
            "tool_usage_counts": self.tool_usage_counts,
        }
