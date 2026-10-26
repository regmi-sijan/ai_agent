import os
import shutil
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

def get_next_run_id(base_dir: Optional[Path] = None) -> str:
    """Generate the next sequential run number: run_001, run_002, run_003, etc."""
    if not base_dir:
        base_dir = Path(__file__).resolve().parent.parent / "outputs"
    base_dir = Path(base_dir)

    candidate_dirs = [base_dir / "runs", base_dir / "logs"]
    existing_nums = []

    for c_dir in candidate_dirs:
        if c_dir.exists():
            for d in c_dir.iterdir():
                if d.is_dir() and d.name.startswith("run_"):
                    suffix = d.name.replace("run_", "")
                    try:
                        num = int(suffix)
                        existing_nums.append(num)
                    except ValueError:
                        pass

    next_num = max(existing_nums) + 1 if existing_nums else 1
    return f"run_{next_num:03d}"

class RunLogger:
    """Manages dedicated, self-contained run folders for autonomous scientific experiments.
    Every run is given a unique sequential run-number folder (e.g. outputs/runs/run_001/):
      - figures/ (all plots, ROC curves, loss curves generated)
      - models/ (trained model weights, .pt, .pth, .joblib)
      - scripts/ (all Python scripts & ROOT macros used)
      - data/ (archived copy of input datasets)
      - summary.md (what was done, metrics, user steering, peer review)
      - methodology_plan.md (theoretical roadmap)
      - inter_agent_dialogue.md (complete dialogue & steering transcript)
      - results.json (machine-readable structured summary)
    """

    def __init__(self, run_id: Optional[str] = None, base_dir: Optional[Path] = None):
        if not base_dir:
            base_dir = Path(__file__).resolve().parent.parent / "outputs"
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.base_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if not run_id:
            self.run_id = get_next_run_id(self.base_dir)
        elif run_id.startswith("run_") and run_id.replace("run_", "").isdigit():
            self.run_id = run_id
        else:
            self.run_id = get_next_run_id(self.base_dir)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Run-specific sub-folders inside this individual run folder
        self.scripts_dir = self.run_dir / "scripts"
        self.figures_dir = self.run_dir / "figures"
        self.models_dir = self.run_dir / "models"
        self.data_dir = self.run_dir / "data"
        for d in [self.scripts_dir, self.figures_dir, self.models_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Link to legacy logs dir for seamless backward compatibility
        legacy_link = self.logs_dir / self.run_id
        if not legacy_link.exists():
            try:
                legacy_link.symlink_to(self.run_dir, target_is_directory=True)
            except Exception:
                pass

        self.dialogue_history: List[Dict[str, str]] = []
        self.user_directives: List[Dict[str, str]] = []
        self.start_time = time.time()
        self.metadata: Dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "start_time": self.start_time,
        }

    def archive_dataset(self, dataset_path: str):
        """Copy input datasets directly into this run's data/ sub-folder."""
        if not dataset_path:
            return
        for p_str in dataset_path.split(","):
            p = Path(p_str.strip())
            if p.exists() and p.is_file():
                try:
                    dest = self.data_dir / p.name
                    if not dest.exists() and p.stat().st_size <= 200 * 1024 * 1024:
                        shutil.copy2(p, dest)
                except Exception:
                    pass
            elif p.exists() and p.is_dir():
                for f in p.glob("*"):
                    if f.is_file() and f.stat().st_size <= 100 * 1024 * 1024:
                        try:
                            shutil.copy2(f, self.data_dir / f.name)
                        except Exception:
                            pass

    def log_dialogue(self, sender: str, recipient: str, message: str):
        """Record inter-agent dialogue message."""
        self.dialogue_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "sender": sender,
            "recipient": recipient,
            "message": message
        })
        self._write_dialogue_file()

    def log_user_directive(self, directive: str):
        """Record a live user steering directive."""
        self.user_directives.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "directive": directive
        })
        self.log_dialogue("User Steering (Live)", "Agents", directive)

    def _write_dialogue_file(self):
        file_path = self.run_dir / "inter_agent_dialogue.md"
        lines = [f"# Inter-Agent Dialogue & Live Steering Log: {self.run_id}\n\n"]
        for item in self.dialogue_history:
            lines.append(f"### `[{item['timestamp']}]` **{item['sender']}** ➔ **{item['recipient']}**\n\n{item['message']}\n\n---\n\n")
        file_path.write_text("".join(lines), encoding="utf-8")

    def sync_generated_artifacts(self):
        """Ensure all scripts, figures, and models created or modified during this run are captured."""
        global_scripts = self.base_dir / "scripts"
        global_figures = self.base_dir / "figures"
        global_models = self.base_dir / "models"

        # Capture any artifacts written to global directories
        for g_dir, run_sub in [(global_scripts, self.scripts_dir), (global_figures, self.figures_dir), (global_models, self.models_dir)]:
            if g_dir.exists():
                for f in g_dir.glob("*"):
                    if f.is_file() and f.stat().st_mtime >= self.start_time - 2:
                        dest = run_sub / f.name
                        if not dest.exists():
                            shutil.copy2(f, dest)

        # Mirror run figures into global figures directory for immediate live GUI viewing
        if self.figures_dir.exists() and global_figures.exists():
            for f in self.figures_dir.glob("*"):
                if f.is_file():
                    dest = global_figures / f.name
                    if not dest.exists() or dest.stat().st_mtime < f.stat().st_mtime:
                        try:
                            shutil.copy2(f, dest)
                        except Exception:
                            pass

    def finalize_run(
        self,
        dataset_path: str,
        task: str,
        target_precision: Optional[str],
        status: str,
        methodology_plan: str,
        final_result: str,
        peer_review: Optional[str] = None,
        iterations: int = 1
    ) -> Path:
        """Compile and write summary.md, methodology_plan.md, and results.json."""
        elapsed = round(time.time() - self.start_time, 1)

        # 1. Write methodology plan
        (self.run_dir / "methodology_plan.md").write_text(methodology_plan or "(No plan recorded)", encoding="utf-8")

        # 2. Sync all artifacts generated during run
        self.sync_generated_artifacts()

        # 3. Archive input datasets into run data directory if reasonable size
        data_dir = self.run_dir / "data"
        if dataset_path:
            for p_str in dataset_path.split(","):
                p = Path(p_str.strip())
                if p.exists() and p.is_file():
                    data_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        # Copy dataset file if under 100MB
                        if p.stat().st_size <= 100 * 1024 * 1024:
                            shutil.copy2(p, data_dir / p.name)
                    except Exception:
                        pass

        saved_scripts = sorted([f.name for f in self.scripts_dir.glob("*") if f.is_file()])
        saved_figures = sorted([f.name for f in self.figures_dir.glob("*") if f.is_file()])
        saved_models = sorted([f.name for f in self.models_dir.glob("*") if f.is_file()])

        # 3. Write structured results.json
        results_data = {
            "run_id": self.run_id,
            "timestamp": self.metadata["timestamp"],
            "elapsed_seconds": elapsed,
            "status": status,
            "dataset_path": dataset_path,
            "task": task,
            "target_precision": target_precision,
            "iterations": iterations,
            "user_directives_count": len(self.user_directives),
            "user_directives": self.user_directives,
            "scripts": saved_scripts,
            "figures": saved_figures,
            "models": saved_models,
            "peer_review_present": bool(peer_review),
        }
        (self.run_dir / "results.json").write_text(json.dumps(results_data, indent=2), encoding="utf-8")

        # 4. Compile comprehensive summary.md
        summary_md = f"""# 🔬 Autonomous Run Summary & Scientific Log

* **Run Directory:** `{self.run_dir}`
* **Run ID:** `{self.run_id}`
* **Timestamp:** `{self.metadata['timestamp']}`
* **Duration:** `{elapsed}s`
* **Pipeline Status:** `{status}`
* **Dataset(s):** `{dataset_path}`
* **Target Precision:** `{target_precision or 'Highest achievable precision'}`

---

## 🎯 Task Objective & Physics Goal
{task}

---

## ⚡ Live User Steering Directives Applied
"""
        if self.user_directives:
            for ud in self.user_directives:
                summary_md += f"- **`[{ud['timestamp']}]`** {ud['directive']}\n"
        else:
            summary_md += "*No mid-run user steering directives were injected for this run.*\n"

        summary_md += f"""
---

## 📊 Experimental Results & Final Metrics
{final_result or '(No result output recorded)'}

---

## ⚖️ Formal Peer Review Referee Evaluation
{peer_review or '(No formal review recorded)'}

---

## 📁 Archived Run Artifacts

* **Executed Code Scripts ({len(saved_scripts)}):**
"""
        if saved_scripts:
            for s in saved_scripts:
                summary_md += f"  - [`scripts/{s}`](scripts/{s})\n"
        else:
            summary_md += "  *(No standalone scripts saved)*\n"

        summary_md += f"\n* **Generated Figures ({len(saved_figures)}):**\n"
        if saved_figures:
            for fg in saved_figures:
                summary_md += f"  - [`figures/{fg}`](figures/{fg})\n"
        else:
            summary_md += "  *(No figures generated)*\n"

        if saved_models:
            summary_md += f"\n* **Saved Models ({len(saved_models)}):**\n"
            for m in saved_models:
                summary_md += f"  - [`models/{m}`](models/{m})\n"

        summary_md += f"""
---

## 🔗 Related Files
* **Full Inter-Agent Dialogue:** [`inter_agent_dialogue.md`](inter_agent_dialogue.md)
* **Full Methodology Plan:** [`methodology_plan.md`](methodology_plan.md)
* **Machine-Readable Metadata:** [`results.json`](results.json)
"""
        (self.run_dir / "summary.md").write_text(summary_md, encoding="utf-8")

        # 5. Update latest_run.txt pointer
        try:
            (self.logs_dir / "latest_run.txt").write_text(str(self.run_dir), encoding="utf-8")
        except Exception:
            pass

        return self.run_dir
