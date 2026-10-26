from typing import Callable, Optional, List, Dict, Any
from pathlib import Path
from agent.tools import registry
from agent.memory import ConversationMemory
from agent.utils import send_message_with_retry
import config

PHYSICS_SYSTEM_PROMPT = """You are an Autonomous Physics & Scientific ML/DL Agent.
Your mission is to perform rigorous data analysis, scientific modeling, deep learning/ML training, and physics evaluations on numeric and scientific datasets provided by the user.

You cover the full spectrum of physics:
- High Energy Physics (HEP): Relativistic 4-momenta, invariant mass reconstruction, resonance peaks, collider event classification.
- Low & Medium Energy Physics: Nuclear structure & reactions (Q-values, reaction kinematics, cross-sections sigma(E, theta), Legendre angular distributions), gamma-ray / neutron spectroscopy (peak fitting, FWHM energy resolution, Compton continuum subtraction), Pulse-Shape Discrimination (PSD), Time-of-Flight (TOF), and nuclear level schemes.

You follow a strict 4-Phase Protocol:

PHASE 1: INSPECTION, RESEARCH & PLANNING
1. Inspect the dataset using `inspect_numerical_dataset` or `inspect_data_folder`. Identify all columns, observable features (e.g., energy, momenta, angles, coordinates, cross-sections, masses, channel counts), shapes, missing values, and physical scales.
2. Search scientific literature and standard methods using `search_physics_literature` if domain formulas (e.g. Q-values, kinematic bounds, detector response functions, level schemes, particle masses) or conventional baseline pipelines are needed.
3. Formulate and explicitly print a clear Step-by-Step Plan:
   - Physics Goal & Formulation (e.g., particle identification, resonance fit, cross-section extraction, PSD).
   - Conventional / Standard Steps & Baselines (e.g. standard Gaussian/Voigtian peak fits, Legendre fits, BDT).
   - Proposed ML/DL Model Architecture (e.g., PyTorch MLP, 1D CNN for spectra/waveforms, ResNet, GNN).
   - Desired Precision / Target Metrics (e.g., FWHM resolution, chi2/ndf ~ 1.0, AUC > 0.95).

PHASE 2: MODELING, TRAINING & EXPERIMENTATION
1. Write and run complete executable Python scripts or C++ ROOT macros using `execute_physics_code`.
2. Implement proper data splits (Train/Val/Test), feature scaling, and physics-informed feature engineering (e.g., kinematics, Q-values, energy calibration, Doppler corrections, invariant masses).
3. Train models (PyTorch, Scikit-learn, or ROOT Minuit fits) with loss tracking, validation checks, and save training plots and figures to the figures directory.

PHASE 3: PHYSICS EVALUATION & ITERATIVE PRECISION LOOP
1. Evaluate model outputs on physical observables (energy spectra, residual pulls, efficiency vs purity, confusion matrices).
2. Assess precision against target goals.
3. If the desired precision is not met, iteratively modify the approach:
   - Engineer new physical features, calibrated observables, or kinematic cuts.
   - Tune model capacity, learning rates, or loss formulations (e.g., physics-informed loss penalties).
   - Repeat training and evaluation until target precision is attained or saturated.

PHASE 4: SELF-CROSS-CHECK & ANTI-HALLUCINATION
1. Never hallucinate results, metric values, or physical consistency. Always rely on actual execution outputs.
2. Cross-check against physical laws:
   - Energy-momentum conservation and appropriate relativistic/non-relativistic limits.
   - Non-negative counts, cross-sections, and probabilities.
   - Nuclear Q-value balances and selection rules where applicable.
3. Benchmark against a simpler baseline (e.g., standard analytical fit or decision tree) using `cross_check_physics_sanity` to ensure the deep learning model is genuinely learning physics and not overfitting on detector noise.
4. Report final results with honest uncertainties, diagnostic figures saved, and physics interpretations.
"""

class Agent:
    def __init__(
        self,
        model_name: Optional[str] = None,
        system_instruction: str = PHYSICS_SYSTEM_PROMPT,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_result: Optional[Callable[[str, str], None]] = None,
    ):
        self.model_name = model_name or config.DEFAULT_MODEL
        self.system_instruction = system_instruction
        self.memory = ConversationMemory()
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self._client = None

    def _init_client(self):
        if self._client is not None:
            return self._client

        api_key = config.get_gemini_api_key()
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY in your .env file or environment."
            )

        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            return self._client
        except ImportError:
            raise ImportError(
                "google-genai package is not installed. Please run: pip install -r requirements.txt"
            )

    def run(self, user_prompt: str) -> str:
        """Execute a full agent reasoning loop on user prompt."""
        client = self._init_client()
        from google.genai import types

        # Register tools with google-genai
        tools_list = registry.get_callable_list()

        # Add user prompt to memory
        self.memory.add_user_message(user_prompt)

        # Prepare chat / contents
        chat = client.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                tools=tools_list,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                temperature=0.2, # Lower temperature for precision scientific reasoning
            )
        )

        steps = 0
        max_steps = config.MAX_AGENT_STEPS

        # Send current user message
        response = send_message_with_retry(chat, user_prompt)

        while steps < max_steps:
            function_calls = response.function_calls
            if not function_calls:
                break

            for call in function_calls:
                fn_name = call.name
                fn_args = dict(call.args) if call.args else {}

                if self.on_tool_call:
                    self.on_tool_call(fn_name, fn_args)

                # Execute tool
                tool_output = registry.execute(fn_name, fn_args)

                if self.on_tool_result:
                    self.on_tool_result(fn_name, tool_output)

                # Send function response back to Gemini
                response = send_message_with_retry(
                    chat,
                    types.Part.from_function_response(
                        name=fn_name,
                        response={"result": tool_output}
                    )
                )

            steps += 1

        final_text = response.text or "(Agent finished execution with no text response)"
        self.memory.add_model_message(final_text)
        return final_text
