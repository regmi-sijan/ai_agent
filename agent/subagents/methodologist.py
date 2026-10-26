import json
from typing import Callable, Optional, Dict, Any, List
from agent.tools import registry
from agent.utils import ResilientChatSession, send_message_with_retry
import config

METHODOLOGIST_SYSTEM_PROMPT = """You are the Lead Theoretical & Methodological Physicist sub-agent.
You hold the highest standard of physical rigor, domain theory, and concise, actionable methodologies.

YOUR CAPABILITIES & PROTOCOL:
1. TARGETED RESEARCH:
   - If physical constants, particle masses, or standard formulas are needed, use `simple_web_search` for a quick, concise check. Avoid redundant searches.
2. COMPREHENSIVE ROADMAP DESIGN:
   - State the formal physics formulation (decay kinematics, invariant mass formulas, reaction equations).
   - Formulate derived physical observables (invariant masses, transverse momentum pT, opening angles, energy asymmetry).
   - Impose physical boundary conditions and conservation laws.
   - Prescribe baseline benchmark models vs Deep Learning models.
   - Define clear precision targets (e.g. AUC >= 0.90, peak resolution sigma).
3. ACTIVE CROSS-TALK WITH EXPERIMENTER:
   - Answer consultations from the Experimenter with precise physical formulas and advice.
4. RIGOROUS PEER REVIEW:
   - Evaluate metrics, mass resolution, and sanity checks before formal approval.
"""

class MethodologistAgent:
    def __init__(
        self,
        model_name: Optional[str] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        get_user_instructions: Optional[Callable[[], str]] = None,
    ):
        self.model_name = model_name or config.DEFAULT_MODEL
        self.on_log = on_log
        self.on_tool_call = on_tool_call
        self.get_user_instructions = get_user_instructions
        self._client = None
        # Streamlined, token-efficient tools
        self.allowed_tools = [
            "simple_web_search",
            "inspect_numerical_dataset",
            "inspect_data_folder",
            "inspect_pretrained_model",
            "read_file"
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

    def plan(self, dataset_path: str, user_description: str, target_precision: Optional[str] = None) -> str:
        """Formulate initial physics plan and methodology with deep literature search."""
        client = self._init_client()

        prompt = f"""[SCIENTIFIC TASK BRIEF]
Dataset Path: {dataset_path}
User Description & Goal: {user_description}
Target Precision / Criteria: {target_precision or "Attain highest realistic scientific precision and cross-check baselines"}

MANDATORY PROTOCOL:
1. Inspect the dataset at '{dataset_path}' using `inspect_numerical_dataset` or `inspect_data_folder`.
2. If specific physical benchmarks or formulas are needed, use `simple_web_search`.
3. Formulate the concise, actionable scientific methodology and roadmap for the Experimenter Agent, citing the physical formulas.
"""
        return self._run_reasoning_loop(client, prompt)

    def answer_consultation(self, question: str, context: str = "") -> str:
        """Answer a theoretical or physical consultation question from the Experimenter mid-experiment."""
        client = self._init_client()
        prompt = f"""[MID-EXPERIMENT CONSULTATION FROM EXPERIMENTER]
The Experimenter is currently executing code/training and requests theoretical guidance:
Question: {question}

Current Experimental Context / Observation:
{context or 'During model training / physics evaluation'}

Provide clear, concise physical guidance, formulas, benchmark values, or cut recommendations to help the Experimenter succeed. Use `simple_web_search` only if a specific benchmark lookup is needed.
"""
        return self._run_reasoning_loop(client, prompt)

    def peer_review(self, current_plan: str, experiment_results: str) -> Dict[str, Any]:
        """Conduct a formal peer-review on the Experimenter's output."""
        client = self._init_client()

        prompt = f"""[FORMAL PEER REVIEW OF EXPERIMENTAL RUN]
Proposed Methodology:
{current_plan}

Experimenter Results, Metrics & Sanity Check:
{experiment_results}

Evaluate the experimental run with the rigor of a scientific journal referee:
1. Physical Plausibility: Are conservation laws, non-negative bounds, and physical expectations satisfied?
2. Statistical Rigor: Did the model outperform simple baselines? Are pulls/residuals consistent?
3. Goal Fulfillment: Was the target precision achieved?

Conclude your review with an explicit block:
=== PEER REVIEW DECISION ===
DECISION: [APPROVED or REVISION_REQUIRED]
PHYSICS_SCORE: [0-100]
CRITIQUE: (concise explanation of strengths or what needs revision)
"""
        review_text = self._run_reasoning_loop(client, prompt)
        approved = "DECISION: APPROVED" in review_text.upper() or "DECISION: [APPROVED]" in review_text.upper()
        return {
            "approved": approved,
            "review": review_text
        }

    def revise_plan(self, current_plan: str, experiment_results: str, bottleneck_notes: str) -> str:
        """Revise methodology based on experiment performance and peer-review critiques."""
        client = self._init_client()

        prompt = f"""[EXPERIMENT ITERATION & REVISION DIRECTIVE]
Previous Methodology:
{current_plan}

Experimenter Results & Metrics:
{experiment_results}

Identified Deficiencies & Peer Review Notes:
{bottleneck_notes}

The current results did not reach the required precision or failed physical sanity checks.
Please search arXiv / literature for alternative parameterizations or techniques, diagnose the exact failure point, and provide an updated, refined methodology (e.g. modify feature transformations, loss penalties, physical cuts, or network architecture).
"""
        return self._run_reasoning_loop(client, prompt)

    def _run_reasoning_loop(self, client, prompt: str) -> str:
        from google.genai import types

        session = ResilientChatSession(
            client=client,
            system_instruction=METHODOLOGIST_SYSTEM_PROMPT,
            tools=self._get_callables(),
            temperature=0.2,
            on_log=self.on_log,
            models=[self.model_name, *[m for m in config.EFFICIENT_MODEL_ARCHITECTURES if m != self.model_name]],
        )

        # Prepend any pending user instructions
        pending_directives = self.get_user_instructions() if self.get_user_instructions else ""
        if pending_directives:
            prompt = f"[USER LIVE DIRECTIVE INJECTED]:\n{pending_directives}\n\n" + prompt

        response = session.send_message(prompt)
        steps = 0
        max_steps = getattr(config, "MAX_AGENT_STEPS", 5)

        while steps < max_steps:
            if not response.function_calls:
                break

            for call in response.function_calls:
                fn_name = call.name
                fn_args = dict(call.args) if call.args else {}

                if self.on_tool_call:
                    self.on_tool_call(fn_name, fn_args)

                tool_output = registry.execute(fn_name, fn_args)

                # Check if a user directive arrived while tool was running
                mid_directive = self.get_user_instructions() if self.get_user_instructions else ""
                if mid_directive:
                    tool_output += f"\n\n[USER LIVE DIRECTIVE RECEIVED MID-RUN]:\n{mid_directive}\nPlease incorporate this guidance immediately into your methodology!"
                    if self.on_log:
                        self.on_log(f"⚡ Injected live user directive into Methodologist: '{mid_directive}'")

                response = session.send_message(
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": tool_output}
                    )
                )

            steps += 1

        return response.text or "(Methodologist finished with no plan text)"
