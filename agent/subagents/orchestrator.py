import re
from typing import Callable, Optional, Dict, Any, List
from pathlib import Path
from agent.subagents.methodologist import MethodologistAgent
from agent.subagents.experimenter import ExperimenterAgent
from agent.tools.physics_tools import set_theorist_consult_handler, set_active_run_context
from agent.run_logger import RunLogger

class AutonomousPhysicsOrchestrator:
    """Orchestrates an autonomous closed-loop between:
    1. MethodologistAgent (Theory, arXiv API, Literature search, Planning, Peer Review, Revision)
    2. ExperimenterAgent (Code execution, ML/DL Training, Testing, Physics evaluation, Sanity cross-checks)
    Features real-time bidirectional cross-talk, mid-run user steering, and dedicated run logging.
    """

    def __init__(
        self,
        max_iterations: int = 3,
        run_id: Optional[str] = None,
        on_subagent_start: Optional[Callable[[str, str], None]] = None,
        on_subagent_end: Optional[Callable[[str, str], None]] = None,
        on_crosstalk: Optional[Callable[[str, str, str], None]] = None, # (sender, recipient, message)
        on_tool_call: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        on_tool_result: Optional[Callable[[str, str, str], None]] = None,
    ):
        self.max_iterations = max_iterations
        self.on_subagent_start = on_subagent_start
        self.on_subagent_end = on_subagent_end
        self.on_crosstalk = on_crosstalk
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result

        # Mid-run user steering queue
        self.pending_user_instructions: List[str] = []
        self.all_user_instructions: List[str] = []

        # Run-specific self-contained logger
        self.run_logger = RunLogger(run_id=run_id)

        self.methodologist = MethodologistAgent(
            on_log=self._handle_agent_log,
            on_tool_call=lambda name, args: self._notify_tool("Methodologist", name, args),
            get_user_instructions=self.get_pending_user_instructions
        )
        self.experimenter = ExperimenterAgent(
            on_log=self._handle_agent_log,
            on_tool_call=lambda name, args: self._notify_tool("Experimenter", name, args),
            on_tool_result=lambda name, res: self._notify_result("Experimenter", name, res),
            get_user_instructions=self.get_pending_user_instructions
        )

        # Wire up bidirectional cross-talk between Experimenter and Theorist
        set_theorist_consult_handler(self._handle_theorist_consultation)

    def _handle_agent_log(self, message: str):
        """Handle internal agent and failover logs: record to run log file and update live UI."""
        if self.run_logger:
            self.run_logger.log_dialogue("System Auto-Failover", "Active Agents", message)
        if getattr(self, "on_crosstalk", None):
            self.on_crosstalk("System Auto-Failover", "Active Agents", message)

    def inject_user_instruction(self, instruction: str):
        """Inject a live mid-run instruction from the user directly into the active agents."""
        clean_text = instruction.strip()
        if not clean_text:
            return
        self.pending_user_instructions.append(clean_text)
        self.all_user_instructions.append(clean_text)
        if self.run_logger:
            self.run_logger.log_user_directive(clean_text)
        if self.on_crosstalk:
            self.on_crosstalk("User Steering (Live)", "All Agents", f"⚡ **User Directive:** {clean_text}")

    def get_pending_user_instructions(self) -> str:
        """Fetch and clear any pending user instructions queued for the agents."""
        # Check if user wrote to outputs/checkpoints/live_directive.txt file
        directive_file = Path(__file__).resolve().parent.parent.parent / "outputs" / "checkpoints" / "live_directive.txt"
        if directive_file.exists():
            try:
                content = directive_file.read_text(encoding="utf-8").strip()
                if content:
                    self.pending_user_instructions.append(content)
                    self.all_user_instructions.append(content)
                    if self.run_logger:
                        self.run_logger.log_user_directive(content)
                    directive_file.unlink() # Delete after reading
            except Exception:
                pass

        if not self.pending_user_instructions:
            return ""
        items = list(self.pending_user_instructions)
        self.pending_user_instructions.clear()
        return "\n".join(f"- {msg}" for msg in items)

    def _handle_theorist_consultation(self, question: str, context: str) -> str:
        msg = f"**Question:** {question}\n\n*Context:* {context}"
        if self.run_logger:
            self.run_logger.log_dialogue("Experimenter", "Methodologist", msg)
        if self.on_crosstalk:
            self.on_crosstalk("Experimenter", "Methodologist", msg)

        if self.on_subagent_start:
            self.on_subagent_start("Methodologist", f"Addressing consultation from Experimenter: '{question}'")

        answer = self.methodologist.answer_consultation(question, context)

        ans_msg = f"**Theorist Recommendation:**\n{answer}"
        if self.run_logger:
            self.run_logger.log_dialogue("Methodologist", "Experimenter", ans_msg)
        if self.on_crosstalk:
            self.on_crosstalk("Methodologist", "Experimenter", ans_msg)

        if self.on_subagent_end:
            self.on_subagent_end("Methodologist", f"**Theorist Consultation Reply:**\n{answer}")

        return answer

    def _notify_tool(self, agent_name: str, tool_name: str, args: Dict[str, Any]):
        if self.on_tool_call:
            self.on_tool_call(agent_name, tool_name, args)

    def _notify_result(self, agent_name: str, tool_name: str, result: str):
        if self.on_tool_result:
            self.on_tool_result(agent_name, tool_name, result)

    def run_pipeline(
        self,
        dataset_path: str,
        user_description: str,
        target_precision: Optional[str] = None,
        pretrained_model_path: Optional[str] = None,
        existing_plan: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the end-to-end autonomous research & experiment loop."""
        # Set active run context so all scripts, plots, and models are written directly inside this run folder
        set_active_run_context(self.run_logger.run_dir)
        self.run_logger.archive_dataset(dataset_path)

        iteration = 1
        history: List[Dict[str, Any]] = []

        desc_with_model = user_description
        if pretrained_model_path:
            desc_with_model += f"\n[PRE-TRAINED MODEL]: Checkpoint provided at '{pretrained_model_path}'. Use `inspect_pretrained_model` to examine its architecture and design a transfer learning / fine-tuning strategy."

        # -------------------------------------------------------------
        # Step 1: Deep Literature Search & Planning (or Resume from Checkpoint)
        # -------------------------------------------------------------
        from pathlib import Path
        checkpoint_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        plan_checkpoint_file = checkpoint_dir / "latest_plan.md"

        if existing_plan and existing_plan.strip():
            current_plan = existing_plan.strip()
            if self.on_subagent_start:
                self.on_subagent_start("Methodologist", f"Resuming from verified methodology plan for '{dataset_path}'")
            if self.on_subagent_end:
                self.on_subagent_end("Methodologist", current_plan)
        elif plan_checkpoint_file.exists() and plan_checkpoint_file.stat().st_size > 50 and "resume" in user_description.lower():
            current_plan = plan_checkpoint_file.read_text(encoding="utf-8")
            if self.on_subagent_start:
                self.on_subagent_start("Methodologist", f"Loaded existing plan checkpoint from disk for '{dataset_path}'")
            if self.on_subagent_end:
                self.on_subagent_end("Methodologist", current_plan)
        else:
            if self.on_subagent_start:
                self.on_subagent_start("Methodologist", f"Searching arXiv and formulating scientific methodology for '{dataset_path}'")

            current_plan = self.methodologist.plan(
                dataset_path=dataset_path,
                user_description=desc_with_model,
                target_precision=target_precision
            )

            if self.on_subagent_end:
                self.on_subagent_end("Methodologist", current_plan)

        # Persist latest plan to checkpoint
        try:
            plan_checkpoint_file.write_text(current_plan, encoding="utf-8")
        except Exception:
            pass

        if self.run_logger:
            self.run_logger.log_dialogue("Methodologist", "Experimenter", f"**Proposed Scientific Protocol & Targets:**\n{current_plan}")
        if self.on_crosstalk:
            self.on_crosstalk("Methodologist", "Experimenter", f"**Proposed Scientific Protocol & Targets:**\n{current_plan}")

        # -------------------------------------------------------------
        # Autonomous Iteration Loop between Experimenter & Methodologist
        # -------------------------------------------------------------
        while iteration <= self.max_iterations:
            if self.on_subagent_start:
                self.on_subagent_start(
                    "Experimenter",
                    f"[Iteration {iteration}/{self.max_iterations}] Training, testing, and verifying physics model"
                )

            experiment_result = self.experimenter.execute_plan(
                dataset_path=dataset_path,
                methodology_plan=current_plan,
                iteration_number=iteration
            )

            if self.on_subagent_end:
                self.on_subagent_end("Experimenter", experiment_result)

            if self.run_logger:
                self.run_logger.log_dialogue("Experimenter", "Methodologist", f"**Iteration {iteration} Results:**\n{experiment_result}")
            if self.on_crosstalk:
                self.on_crosstalk("Experimenter", "Methodologist", f"**Iteration {iteration} Results Submitted for Peer Review:**\n{experiment_result}")

            # -------------------------------------------------------------
            # Formal Peer Review by Methodologist
            # -------------------------------------------------------------
            if self.on_subagent_start:
                self.on_subagent_start("Methodologist", f"[Iteration {iteration} Peer Review] Refereeing experimental metrics and physical consistency")

            review = self.methodologist.peer_review(
                current_plan=current_plan,
                experiment_results=experiment_result
            )

            if self.on_subagent_end:
                self.on_subagent_end("Methodologist", review["review"])

            if self.run_logger:
                self.run_logger.log_dialogue("Methodologist", "Experimenter", f"**Referee Decision:**\n{review['review']}")
            if self.on_crosstalk:
                self.on_crosstalk("Methodologist", "Experimenter", f"**Referee Decision:**\n{review['review']}")

            iteration_record = {
                "iteration": iteration,
                "plan": current_plan,
                "result": experiment_result,
                "review": review["review"],
                "approved": review["approved"]
            }
            history.append(iteration_record)

            if review["approved"]:
                # Peer reviewed and verified! Finalize dedicated run log folder
                run_log_dir = self.run_logger.finalize_run(
                    dataset_path=dataset_path,
                    task=user_description,
                    target_precision=target_precision,
                    status="SUCCESS_PEER_REVIEW_APPROVED",
                    methodology_plan=current_plan,
                    final_result=experiment_result,
                    peer_review=review["review"],
                    iterations=iteration
                )
                return {
                    "status": "SUCCESS_PEER_REVIEW_APPROVED",
                    "iterations_completed": iteration,
                    "final_plan": current_plan,
                    "final_result": experiment_result,
                    "peer_review": review["review"],
                    "history": history,
                    "run_log_dir": str(run_log_dir)
                }

            # If not approved and more iterations remain, revise methodology
            if iteration < self.max_iterations:
                if self.on_subagent_start:
                    self.on_subagent_start(
                        "Methodologist",
                        f"[Iteration {iteration + 1} Revision] Incorporating referee critiques & adapting model"
                    )

                current_plan = self.methodologist.revise_plan(
                    current_plan=current_plan,
                    experiment_results=experiment_result,
                    bottleneck_notes=review["review"]
                )

                if self.on_subagent_end:
                    self.on_subagent_end("Methodologist", current_plan)

                if self.run_logger:
                    self.run_logger.log_dialogue("Methodologist", "Experimenter", f"**Revised Protocol for Iteration {iteration + 1}:**\n{current_plan}")
                if self.on_crosstalk:
                    self.on_crosstalk("Methodologist", "Experimenter", f"**Revised Protocol for Iteration {iteration + 1}:**\n{current_plan}")

            iteration += 1

        # Completed max iterations
        run_log_dir = self.run_logger.finalize_run(
            dataset_path=dataset_path,
            task=user_description,
            target_precision=target_precision,
            status="COMPLETED_MAX_ITERATIONS",
            methodology_plan=current_plan,
            final_result=experiment_result,
            peer_review=review["review"] if "review" in locals() else None,
            iterations=self.max_iterations
        )
        return {
            "status": "COMPLETED_MAX_ITERATIONS",
            "iterations_completed": self.max_iterations,
            "final_plan": current_plan,
            "final_result": experiment_result,
            "history": history,
            "run_log_dir": str(run_log_dir)
        }
