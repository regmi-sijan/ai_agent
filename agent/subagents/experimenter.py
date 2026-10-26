import json
from typing import Callable, Optional, Dict, Any, List
from agent.tools import registry
from agent.utils import ResilientChatSession, send_message_with_retry
import config

EXPERIMENTER_SYSTEM_PROMPT = """You are the Experimental ML & Scientific Computing Engineer sub-agent.
Your mission is to take the physics methodology and specifications from the Methodologist and autonomously execute the complete training, testing, validation, and sanity cross-check pipeline.

Rules of Operation:
1. Code Execution: Write complete, self-contained Python or C++ ROOT scripts using `execute_physics_code`. Never leave placeholders or pseudocode. Use PyTorch, NumPy, SciPy, Pandas, Matplotlib, or Scikit-learn as needed.
2. Hardware Acceleration Protocol (TPU -> GPU -> CPU):
   Always configure PyTorch compute devices in this strict hierarchy:
   - Level 1 (TPU): Try Google Cloud TPU via `torch_xla` if available.
   - Level 2 (GPU): If TPU is not available, check for NVIDIA CUDA (`torch.cuda.is_available()`) or Apple Silicon Metal (`torch.backends.mps.is_available()`).
   - Level 3 (CPU): If neither TPU nor GPU is available, fall back cleanly to CPU (`torch.device('cpu')`).
   Include this resilient helper in PyTorch training scripts:
   ```python
   def get_device():
       try:
           import torch_xla.core.xla_model as xm
           return xm.xla_device()
       except Exception:
           pass
       if torch.cuda.is_available():
           return torch.device('cuda')
       if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
           return torch.device('mps')
       return torch.device('cpu')
   device = get_device()
   ```
3. Inter-Agent Cross-Talk: If you encounter unexpected data anomalies, ambiguous peaks, or need advice on physical cuts or formulas while working, call the `consult_theorist` tool. The Methodologist will review the issue (and search literature if needed) and advise you immediately!
3. Artifacts: Ensure all evaluation plots (loss curves, ROC curves, mass distributions, confusion matrices) are saved to `outputs/figures/`.
4. Validation & Sanity Cross-Check:
   - Always run a baseline benchmark (e.g. Random Forest, Logistic Regression, or simple physical cut).
   - Verify physical boundary constraints and conservation laws using `cross_check_physics_sanity`.
5. Result Evaluation:
   - Check if the achieved metrics meet the target precision requested in the plan.
6. Output Formatting:
   At the end of your execution, you MUST output a structured summary section:
   --- EXPERIMENT SUMMARY ---
   STATUS: [TARGET_ACHIEVED or NEEDS_REVISION]
   ACHIEVED_METRICS: (list key metrics e.g. AUC: 0.97, Test Loss: 0.042, Resolution: 12.3 MeV)
   SANITY_CHECK: [PASS or FAIL] (details of boundary/baseline check)
   GENERATED_PLOTS: (list of saved figure files)
   DIAGNOSIS: (if not achieved, what was the bottleneck: underfitting, overfitting, unstable loss, bad cuts?)
"""

class ExperimenterAgent:
    def __init__(
        self,
        model_name: Optional[str] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_result: Optional[Callable[[str, str], None]] = None,
        get_user_instructions: Optional[Callable[[], str]] = None,
    ):
        self.model_name = model_name or config.DEFAULT_MODEL
        self.on_log = on_log
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.get_user_instructions = get_user_instructions
        self._client = None
        self.allowed_tools = [
            "get_optimal_compute_device",
            "execute_physics_code",
            "cross_check_physics_sanity",
            "inspect_numerical_dataset",
            "inspect_data_folder",
            "inspect_pretrained_model",
            "consult_theorist",
            "read_file",
            "write_file",
        ]

    def _init_client(self):
        if self._client is not None:
            return self._client
        api_key = config.get_gemini_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        from google import genai
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _get_callables(self) -> List[Callable]:
        callables = []
        for name in self.allowed_tools:
            tool = registry.get_tool(name)
            if tool:
                callables.append(tool)
        return callables

    def execute_plan(
        self,
        dataset_path: str,
        methodology_plan: str,
        iteration_number: int = 1
    ) -> str:
        """Execute the scientific training and validation pipeline."""
        client = self._init_client()
        from google.genai import types

        prompt = f"""[EXECUTION RUN - ITERATION {iteration_number}]
Dataset Path: {dataset_path}

Methodology & Architecture Specifications from Methodologist:
{methodology_plan}

Instructions:
1. Write and execute the full training and validation pipeline in Python using `execute_physics_code`.
2. Save model checkpoints and diagnostic plots to outputs/figures/.
3. Evaluate metrics against the target precision specified in the plan.
4. Run `cross_check_physics_sanity` with baseline comparisons and boundary tests.
5. Conclude with the required '--- EXPERIMENT SUMMARY ---' block with STATUS: [TARGET_ACHIEVED] or [NEEDS_REVISION].
"""
        return self._run_reasoning_loop(client, prompt)

    def _run_reasoning_loop(self, client, prompt: str) -> str:
        from google.genai import types

        session = ResilientChatSession(
            client=client,
            system_instruction=EXPERIMENTER_SYSTEM_PROMPT,
            tools=self._get_callables(),
            temperature=0.2,
            on_log=self.on_log,
            models=[self.model_name, *[m for m in config.EFFICIENT_MODEL_ARCHITECTURES if m != self.model_name]],
        )

        # Prepend any pending user instructions to the initial prompt
        pending_directives = self.get_user_instructions() if self.get_user_instructions else ""
        if pending_directives:
            prompt = f"[USER LIVE DIRECTIVE INJECTED]:\n{pending_directives}\n\n" + prompt

        response = session.send_message(prompt)
        steps = 0
        max_steps = max(getattr(config, "MAX_AGENT_STEPS", 5) + 3, 7)

        while steps < max_steps:
            if not response.function_calls:
                break

            for call in response.function_calls:
                fn_name = call.name
                fn_args = dict(call.args) if call.args else {}

                if self.on_tool_call:
                    self.on_tool_call(fn_name, fn_args)

                tool_output = registry.execute(fn_name, fn_args)

                if self.on_tool_result:
                    self.on_tool_result(fn_name, tool_output)

                # Check if a user instruction was injected while the tool was running
                mid_directive = self.get_user_instructions() if self.get_user_instructions else ""
                if mid_directive:
                    tool_output += f"\n\n[USER LIVE DIRECTIVE RECEIVED MID-RUN]:\n{mid_directive}\nPlease immediately incorporate this directive into your plan/code!"
                    if self.on_log:
                        self.on_log(f"⚡ Injected live user directive into Experimenter: '{mid_directive}'")

                response = session.send_message(
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": tool_output}
                    )
                )

            steps += 1

        return response.text or "(Experimenter finished with no text output)"
