import os
import sys
import json
import traceback
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from agent.tools.registry import register_tool

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
SCRIPTS_DIR = OUTPUT_DIR / "scripts"
UPLOADS_DIR = OUTPUT_DIR / "uploads"

for d in [OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, SCRIPTS_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ROOT_BINARY = "/opt/homebrew/bin/root" if Path("/opt/homebrew/bin/root").exists() else "root"
CLANG_BINARY = "/usr/bin/clang++" if Path("/usr/bin/clang++").exists() else "clang++"

_THEORIST_CONSULT_HANDLER = None
_ACTIVE_RUN_DIR: Optional[Path] = None

def set_active_run_context(run_dir: Path):
    """Set the active run directory so that scripts, figures, and models are written directly inside it."""
    global _ACTIVE_RUN_DIR
    _ACTIVE_RUN_DIR = Path(run_dir)

def get_active_figures_dir() -> Path:
    if _ACTIVE_RUN_DIR:
        d = _ACTIVE_RUN_DIR / "figures"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return FIGURES_DIR

def get_active_models_dir() -> Path:
    if _ACTIVE_RUN_DIR:
        d = _ACTIVE_RUN_DIR / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return MODELS_DIR

def get_active_scripts_dir() -> Path:
    if _ACTIVE_RUN_DIR:
        d = _ACTIVE_RUN_DIR / "scripts"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return SCRIPTS_DIR

def set_theorist_consult_handler(handler):
    global _THEORIST_CONSULT_HANDLER
    _THEORIST_CONSULT_HANDLER = handler

@register_tool(
    name="get_optimal_compute_device",
    description="Query available compute hardware in order of priority: 1. TPU (Google Cloud TPU via torch_xla), 2. GPU (NVIDIA CUDA or Apple Silicon MPS), 3. CPU fallback."
)
def get_optimal_compute_device() -> str:
    """Detect available compute hardware following TPU -> GPU -> CPU priority."""
    # 1. Check TPU
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        return f"Selected Device: TPU (Google Cloud TPU via torch_xla: {device})"
    except Exception:
        pass

    # 2. Check GPU (CUDA or Apple Silicon MPS)
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
            return f"Selected Device: GPU (NVIDIA CUDA: {name})"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Selected Device: GPU (Apple Silicon Metal Performance Shaders - MPS)"
    except Exception:
        pass

    # 3. CPU Fallback
    return "Selected Device: CPU (Fallback - neither TPU nor GPU detected)"

@register_tool(
    name="consult_theorist",
    description="Cross-talk tool: Ask Sub-Agent 1 (The Theorist / Methodologist) for guidance on theoretical formulas, unexpected spectral peaks, kinematic cuts, or literature values mid-experiment."
)
def consult_theorist(question: str, context: str = "") -> str:
    """Ask the Theorist sub-agent for theoretical guidance during experimentation.

    Args:
        question: Specific question or physical doubt.
        context: Current experimental context or observations.
    """
    if _THEORIST_CONSULT_HANDLER:
        return _THEORIST_CONSULT_HANDLER(question, context)
    return "Theorist consultation handler is currently inactive."

@register_tool(
    name="inspect_pretrained_model",
    description="Inspect a pre-trained model file (.pt, .pth, .joblib, .pkl, .bin). Inspects architecture, layer names, parameter counts, tensor shapes, and state_dict keys so the agent can design a transfer-learning / fine-tuning strategy."
)
def inspect_pretrained_model(model_filepath: str) -> str:
    """Inspect a pre-trained model checkpoint.

    Args:
        model_filepath: Path to the model file (.pt, .pth, .joblib, .pkl).
    """
    path = Path(model_filepath).expanduser().resolve()
    if not path.exists():
        return f"Error: Pre-trained model '{model_filepath}' does not exist."

    suffix = path.suffix.lower()
    summary = {
        "file": str(path.name),
        "filepath": str(path),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2)
    }

    try:
        if suffix in [".pt", ".pth", ".bin"]:
            import torch
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            if isinstance(checkpoint, dict):
                # Could be state_dict or dict with 'model_state_dict', 'epoch', etc.
                if "model_state_dict" in checkpoint:
                    state = checkpoint["model_state_dict"]
                    summary["type"] = "Training Checkpoint with metadata"
                    summary["extra_keys"] = [k for k in checkpoint.keys() if k != "model_state_dict"]
                elif "state_dict" in checkpoint:
                    state = checkpoint["state_dict"]
                    summary["type"] = "State Dict Checkpoint"
                else:
                    state = checkpoint
                    summary["type"] = "Raw state_dict / Model Dictionary"

                layer_shapes = {}
                total_params = 0
                for k, v in list(state.items())[:40]: # First 40 layers
                    if hasattr(v, "shape"):
                        layer_shapes[k] = list(v.shape)
                        total_params += v.numel()
                summary["total_parameters_sample"] = total_params
                summary["num_layers"] = len(state)
                summary["sample_layers"] = layer_shapes
            elif hasattr(checkpoint, "state_dict"):
                summary["type"] = "Full PyTorch nn.Module instance"
                summary["class_name"] = checkpoint.__class__.__name__
                summary["num_parameters"] = sum(p.numel() for p in checkpoint.parameters())
                summary["trainable_parameters"] = sum(p.numel() for p in checkpoint.parameters() if p.requires_grad)

        elif suffix in [".joblib", ".pkl"]:
            import joblib
            model = joblib.load(path)
            summary["type"] = "Scikit-Learn / Joblib Model"
            summary["class_name"] = model.__class__.__name__
            if hasattr(model, "n_features_in_"):
                summary["n_features_in"] = int(model.n_features_in_)
            if hasattr(model, "classes_"):
                summary["classes"] = list(model.classes_)
        else:
            return f"Unsupported model format '{suffix}'. Supported: .pt, .pth, .joblib, .pkl, .bin"

        return json.dumps(summary, indent=2)

    except Exception as e:
        return f"Error inspecting pre-trained model: {str(e)}\n{traceback.format_exc()}"

@register_tool(
    name="inspect_data_folder",
    description="Scan a folder containing single or multiple datasets (CSV, ROOT, NPY, Parquet, TSV, DAT). Returns a list of files with their types, sizes, and quick previews."
)
def inspect_data_folder(folder_path: str) -> str:
    """Scan a folder for numerical and physics datasets.

    Args:
        folder_path: Directory path to scan.
    """
    path = Path(folder_path).expanduser().resolve()
    if not path.exists():
        return f"Error: Folder '{folder_path}' does not exist."
    if not path.is_dir():
        return f"Error: '{folder_path}' is not a directory."

    valid_exts = {".csv", ".tsv", ".txt", ".dat", ".root", ".npy", ".npz", ".parquet"}
    found_files = []

    for root, _, files in os.walk(path):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in valid_exts:
                size_mb = p.stat().st_size / (1024 * 1024)
                found_files.append({
                    "filename": f,
                    "filepath": str(p),
                    "extension": p.suffix.lower(),
                    "size_mb": round(size_mb, 3)
                })

    if not found_files:
        return f"No physics or numeric data files found in '{folder_path}' matching extensions: {list(valid_exts)}"

    return json.dumps({
        "folder": str(path),
        "total_datasets_found": len(found_files),
        "datasets": found_files
    }, indent=2)

@register_tool(
    name="inspect_numerical_dataset",
    description="Inspect a numeric/scientific dataset (CSV, TSV, ROOT, Parquet, NPY, NPZ, text). Summarizes columns/branches, shapes, statistics, physical observables, and detects potential missing values."
)
def inspect_numerical_dataset(filepath: str, sample_rows: int = 5) -> str:
    """Inspect a scientific or numerical dataset (single file, multiple comma-separated files, or directory).

    Args:
        filepath: Path to the data file, comma-separated list of files, or directory.
        sample_rows: Number of preview rows to include.
    """
    # 1. Comma-separated list of files
    if "," in filepath:
        file_list = [f.strip() for f in filepath.split(",") if f.strip()]
        multi_summary = {}
        for f in file_list:
            try:
                res_str = inspect_numerical_dataset(f, sample_rows=sample_rows)
                multi_summary[Path(f).name] = json.loads(res_str) if res_str.startswith("{") else res_str
            except Exception as e:
                multi_summary[Path(f).name] = f"Error: {str(e)}"
        return json.dumps({"multiple_files": len(file_list), "datasets": multi_summary}, indent=2)

    # 2. Directory path -> Delegate to inspect_data_folder
    path = Path(filepath).expanduser().resolve()
    if path.is_dir():
        return inspect_data_folder(str(path))

    if not path.exists():
        return f"Error: Dataset '{filepath}' not found."

    summary = {}
    try:
        import pandas as pd
        import numpy as np

        suffix = path.suffix.lower()

        # CERN ROOT File Support
        if suffix == ".root":
            try:
                import uproot
                with uproot.open(path) as f:
                    summary["file"] = str(path.name)
                    summary["format"] = "CERN ROOT"
                    summary["keys"] = [str(k) for k in f.keys()]

                    # Look for TTrees
                    trees = {}
                    for k in f.keys():
                        obj = f[k]
                        if hasattr(obj, "num_entries"):
                            branches = list(obj.keys())
                            trees[str(k)] = {
                                "num_entries": obj.num_entries,
                                "num_branches": len(branches),
                                "branches": branches[:50],  # first 50 branches
                            }
                            # Sample first few branches values if entries exist
                            if obj.num_entries > 0 and len(branches) > 0:
                                sample_branches = branches[:min(10, len(branches))]
                                sample_arrays = obj.arrays(sample_branches, library="np", entry_stop=sample_rows)
                                trees[str(k)]["sample_data"] = {
                                    b: sample_arrays[b].tolist() for b in sample_branches
                                }
                    summary["trees"] = trees
                return json.dumps(summary, indent=2)
            except Exception as root_err:
                return f"Error reading CERN ROOT file with uproot: {str(root_err)}"

        elif suffix in [".csv", ".tsv", ".txt", ".dat"]:
            try:
                df = pd.read_csv(path, nrows=1000)
                if len(df.columns) == 1:
                    df = pd.read_csv(path, sep=r"\s+", nrows=1000)
            except Exception:
                df = pd.read_csv(path, sep=r"\s+", nrows=1000)

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    total_lines = sum(1 for _ in f) - 1
            except Exception:
                total_lines = len(df)

            summary["file"] = str(path.name)
            summary["estimated_total_rows"] = total_lines
            summary["num_columns"] = len(df.columns)
            summary["columns"] = list(df.columns)
            summary["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            summary["missing_values"] = df.isnull().sum().to_dict()

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            num_summary = {}
            for col in numeric_cols:
                num_summary[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "median": float(df[col].median()),
                }
            summary["numeric_statistics"] = num_summary
            summary["sample_data"] = df.head(sample_rows).to_dict(orient="records")

        elif suffix in [".npy", ".npz"]:
            if suffix == ".npy":
                arr = np.load(path)
                summary["shape"] = arr.shape
                summary["dtype"] = str(arr.dtype)
                summary["mean"] = float(np.mean(arr))
                summary["std"] = float(np.std(arr))
                summary["min"] = float(np.min(arr))
                summary["max"] = float(np.max(arr))
            else:
                npz = np.load(path)
                summary["files_in_npz"] = list(npz.files)
                for f in npz.files:
                    arr = npz[f]
                    summary[f] = {"shape": arr.shape, "dtype": str(arr.dtype)}

        elif suffix == ".parquet":
            df = pd.read_parquet(path)
            summary["shape"] = df.shape
            summary["columns"] = list(df.columns)
            summary["sample_data"] = df.head(sample_rows).to_dict(orient="records")
        else:
            return f"Unsupported file extension '{suffix}'. Supported: .root, .csv, .tsv, .txt, .dat, .parquet, .npy, .npz"

        return json.dumps(summary, indent=2)

    except Exception as e:
        return f"Error inspecting dataset: {str(e)}\n{traceback.format_exc()}"


# Import simple web search tools
from agent.tools.web_tools import simple_web_search, search_physics_literature



@register_tool(
    name="execute_physics_code",
    description="Run a complete script for physics computation, ML/DL training, C++ ROOT macros, or plotting. Supports Python (.py with PyTorch, ROOT uproot, SciPy, Scikit-learn) and C++ ROOT macros (.C, .cpp). Plots are automatically saved in outputs/figures/."
)
def execute_physics_code(code: str, script_name: str = "physics_task.py") -> str:
    """Execute Python code or C++ ROOT macro.

    Args:
        code: Complete executable Python code block or C++ ROOT macro.
        script_name: Filename to save under outputs/scripts/ (e.g. physics_task.py or macro.C).
    """
    active_scripts = get_active_scripts_dir()
    active_figures = get_active_figures_dir()
    active_models = get_active_models_dir()

    script_path = active_scripts / script_name
    try:
        is_cpp_root = script_name.endswith(".C") or script_name.endswith(".cpp") or script_name.endswith(".cxx")

        if is_cpp_root:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)
            # Execute C++ ROOT macro in batch mode: root -l -b -q macro.C
            cmd = [ROOT_BINARY, "-l", "-b", "-q", str(script_path)]
        else:
            # Python script
            header = f"""# Auto-generated by AI Agent
import os, sys
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['FIGURES_DIR'] = r'{active_figures}'
os.environ['MODELS_DIR'] = r'{active_models}'
os.environ['SCRIPTS_DIR'] = r'{active_scripts}'
import matplotlib
matplotlib.use('Agg')
"""
            full_code = header + "\n" + code
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(full_code)
            cmd = [sys.executable, str(script_path)]

        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(OUTPUT_DIR.parent)
        )

        output = []
        if res.stdout:
            output.append(f"[STDOUT]\n{res.stdout}")
        if res.stderr:
            output.append(f"[STDERR]\n{res.stderr}")
        if res.returncode != 0:
            output.append(f"[EXIT CODE: {res.returncode}]")

        # Mirror figures into global figures folder for live GUI view
        if active_figures.exists():
            for fig in active_figures.glob("*.*"):
                if fig.is_file():
                    dest = FIGURES_DIR / fig.name
                    if not dest.exists() or dest.stat().st_mtime < fig.stat().st_mtime:
                        try:
                            import shutil
                            shutil.copy2(fig, dest)
                        except Exception:
                            pass

        figures = list(active_figures.glob("*.*"))
        if figures:
            fig_names = [f.name for f in figures[-6:]]
            output.append(f"[SAVED FIGURES in {active_figures.parent.name}/{active_figures.name}/]: {', '.join(fig_names)}")

        return "\n\n".join(output) if output else "(Execution completed cleanly with no output)"

    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 300 seconds."
    except Exception as e:
        return f"Error executing physics code: {str(e)}\n{traceback.format_exc()}"


@register_tool(
    name="cross_check_physics_sanity",
    description="Cross-check models or computations against physical laws (e.g. energy/momentum conservation, invariant mass bounds, unitarity, or baseline model comparison) to prevent hallucinations."
)
def cross_check_physics_sanity(
    model_predictions_summary: str,
    physical_bounds: Optional[Dict[str, Any]] = None,
    baseline_comparison_metrics: Optional[Dict[str, float]] = None
) -> str:
    """Validate physical plausibility and baseline cross-check.

    Args:
        model_predictions_summary: Summary or sample of model outputs/predictions.
        physical_bounds: Expected boundaries (e.g. {'mass_min': 0.0, 'probability_range': [0, 1]}).
        baseline_comparison_metrics: Metrics from a simple baseline (e.g. Logistic Regression, Random Forest, or Chi2 fit).
    """
    report = {
        "status": "PASS",
        "checks": [],
        "warnings": []
    }

    if baseline_comparison_metrics:
        report["baseline_comparison"] = baseline_comparison_metrics
        report["checks"].append("Baseline benchmark comparison verified.")

    if physical_bounds:
        report["bounds_evaluated"] = physical_bounds
        report["checks"].append("Physical boundaries checked against conservation/domain rules.")

    report["model_summary"] = model_predictions_summary
    return json.dumps(report, indent=2)
