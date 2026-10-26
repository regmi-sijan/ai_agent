# Antigravity Physics Agent

A local, Gemini-powered research assistant for physics data analysis and scientific machine learning. Use a browser dashboard or a terminal to inspect datasets, develop an analysis plan, execute generated Python or CERN ROOT scripts, and review results through a two-agent workflow.

This is an experimental research tool. It executes generated code with your user account's permissions. Run it in a disposable environment with trusted data, and independently check scientific results. The dashboard has no authentication and defaults to `127.0.0.1`; keep it local.

## What it can do

- Inspect CSV, TSV, whitespace-delimited text/DAT, NumPy NPY/NPZ, Parquet, and ROOT datasets; scan folders recursively for supported files.
- Coordinate a **Methodologist** for plans and model-based review with an **Experimenter** for code generation, execution, and evaluation.
- Attempt classification, regression, spectrum fitting, invariant-mass reconstruction, and other analyses described in your prompt using the installed scientific Python stack.
- Inspect local PyTorch or joblib model checkpoints for possible reuse.
- Search short web snippets through DuckDuckGo, with a Wikipedia fallback.
- Stream agent messages and tool results to a FastAPI dashboard, display generated figures, accept live guidance, and browse archived runs.
- Save scripts, figures, models, plans, dialogue, and summaries in numbered run folders.
- Detect TPU, CUDA, Apple MPS, or CPU availability. Generated training code must actually select and use the device.

Tasks and architectures are generated at runtime. There are no bundled pretrained models, fixed validated analysis pipelines, or guaranteed accuracy targets.

## Requirements

- Python 3.11 or newer; Python 3.11/3.12 are recommended starting points for scientific dependency compatibility.
- A Google Gemini API key and access to a model that supports function calling.
- Internet access for Gemini and optional searches. The dashboard also loads styling, icons, Markdown, and equations from third-party CDNs.
- Enough RAM and disk space for your datasets and model training. A GPU is optional.
- Optional: a separate CERN ROOT installation with `root` on `PATH` for C++ macros. Reading ROOT files with `uproot` does not require CERN ROOT.

The project uses `google-genai`, FastAPI, Uvicorn, NumPy, pandas, SciPy, scikit-learn, Matplotlib, PyTorch, uproot, awkward, and PyArrow. See [requirements.txt](requirements.txt) for the installation list.

## Installation

```bash
git clone https://github.com/regmi-sijan/ai_agent.git
cd ai_agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` and copy with `Copy-Item .env.example .env`.

Edit `.env` and set your own `GEMINI_API_KEY`. Set `AGENT_MODEL` to a function-calling model available to your account. The example default is a configuration starting point; model availability and quotas depend on your provider account. Never commit your real `.env`.

## Start the dashboard

```bash
python main.py --gui
```

The browser opens at `http://127.0.0.1:8000`. Use `--port 8080` to choose another port.

1. Upload dataset files, or select **Folder Path** and enter a file or directory accessible to the server.
2. Describe the columns, units, scientific question, and expected outputs in the task field.
3. Optionally provide a trusted local model checkpoint and target criteria.
4. Launch the pipeline. Watch the Methodologist, Experimenter, dialogue, figures, and run archive tabs.
5. Use the live steering field to add guidance at an agent's next reasoning/tool boundary.

The resume control reuses a methodology plan; it does not restore in-memory model training, optimizer state, or an interrupted execution process. Run one pipeline at a time: output routing and consultation handlers use shared global state.

## Terminal usage

```bash
# Inspect and summarize the included toy data.
python main.py --pipeline --data sample_physics_data.csv \
  --task "Inspect the columns, report missing values, and plot the distributions of px, py, pz, and E. Explain why ten rows are insufficient for reliable ML evaluation."

# Multiple files or a directory.
python main.py --pipeline --data signal.csv background.csv \
  --task "Compare the datasets and propose a baseline classifier with a held-out evaluation."
python main.py --pipeline --data /path/to/datasets \
  --task "Inspect the datasets, identify observables and units, and fit the energy spectrum."

# Optional precision criteria and trusted local checkpoint.
python main.py --pipeline --data /path/to/data.csv \
  --task "Evaluate this classifier against a simple baseline." \
  --target "Report ROC AUC with uncertainty" --model /path/to/weights.pt

# Single assistant query, interactive shell, and argument help.
python main.py "Inspect sample_physics_data.csv and describe its columns"
python main.py
python main.py --help
```

Interactive commands: `/tools`, `/help`, `/clear`, `/pipeline <file> <description>`, and `/exit`. The general assistant starts a new model chat for each query; stored message history is currently not replayed to the model.

For CLI live steering, while a pipeline is running:

```bash
mkdir -p outputs/checkpoints
printf '%s\n' 'Compare against a simpler baseline before increasing model complexity.' > outputs/checkpoints/live_directive.txt
```

The directive is consumed at the next supported agent boundary. Dataset filenames containing commas are unsupported by the current multi-file delimiter logic.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | none | Required Gemini API key; `GOOGLE_API_KEY` is also accepted. |
| `AGENT_MODEL` | `gemini-2.5-flash` | Primary function-calling model ID. |
| `AGENT_FALLBACK_MODELS` | empty | Comma-separated model IDs to try when a pipeline model is unavailable, overloaded, or quota-limited. |
| `MAX_AGENT_STEPS` | `5` | Manual tool-loop round limit for the general assistant and Methodologist. The Experimenter uses `max(value + 3, 7)`. |
| `RATE_LIMIT_DELAY` | `2.5` | Minimum spacing in seconds for model requests; adjust for your quota. This is not a quota guarantee. |

Environment variables take precedence over `.env`. Pipeline orchestration allows up to three experiment/review iterations by default. Python/C++ executions have a 300-second timeout; general shell commands have a 30-second timeout. Dependencies required by generated scripts beyond the requirements file must be installed separately.

## Project contents

```text
main.py                       CLI, interactive assistant, dashboard entry point
config.py                     Environment loading and model configuration
agent/core.py                 General assistant and manual tool execution loop
agent/memory.py               Local conversation record
agent/utils.py                Request pacing, retry, and pipeline model fallback
agent/subagents/               Methodologist, Experimenter, and orchestrator
agent/tools/                   Dataset, execution, search, and filesystem tools
agent/run_logger.py            Run IDs, dataset copies, dialogue, summaries
gui/server.py                 Local FastAPI routes and event streaming
gui/templates/index.html      Browser dashboard
sample_physics_data.csv        Ten-row toy dataset for inspecting input structure
.env.example                   Configuration template without credentials
requirements.txt               Python runtime dependencies
requirements-dev.txt           Additional verification dependencies
docs/                         Architecture and usage details
tests/                        Offline smoke and regression checks
```

The ten-row sample is illustrative data, not a validated physics benchmark. Bring your own appropriately licensed, validated datasets and specify units, labels, and expected conventions.

## Generated outputs

```text
outputs/
  runs/run_001/
    data/                      Copies of eligible input files
    scripts/                   Generated Python / ROOT scripts
    figures/                   Diagnostic figures
    models/                    Saved checkpoints when generated
    methodology_plan.md
    inter_agent_dialogue.md
    summary.md
    results.json
  figures/                     Shared live figure gallery
  uploads/                     Browser uploads
  checkpoints/latest_plan.md   Latest saved plan
  logs/                        Compatibility links to numbered runs
```

These directories are created automatically and excluded from Git. Input archiving is size-limited and folder copies are shallow, so run folders are not guaranteed to contain every source dataset. `results.json` records run metadata and artifact lists; scientific metrics are generally in the report text or generated files rather than a fixed metrics schema. Inspect generated scripts for seeds, dependencies, and dataset paths before attempting reproduction.

## Scientific and security limitations

- Agent review is another model's assessment, not independent scientific validation. Requested precision may never be reached.
- `cross_check_physics_sanity` currently records supplied summaries, bounds, and baseline metadata. It does **not** numerically test conservation laws or prediction bounds, even when its output says `PASS`.
- Literature tools return brief snippets. The arXiv-named compatibility wrapper also uses general search; there is no direct arXiv API implementation. Full-page fetching is disabled.
- Python, shell, ROOT code, and checkpoint loaders are not sandboxed. Pickle/joblib and full PyTorch checkpoint loading can execute code. Use only trusted checkpoints.
- Dataset samples, paths, prompts, and tool outputs can be sent to Gemini or search services. Only use data you are authorized to share.
- The dashboard has no authentication, upload limits, session isolation guarantees, or hardened HTML rendering. Do not expose it to a network or use it as a multi-user service.
- There is no cancellation control for an active generated process. Stopping the app may leave a child process that needs separate termination.

Read [usage examples](docs/USAGE.md), [architecture](docs/ARCHITECTURE.md), and [security guidance](SECURITY.md) for more detail.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m compileall -q main.py config.py agent gui
python -m pip check
```

CI runs offline tests without an API key. Live Gemini runs incur provider usage and are not part of CI. See [CONTRIBUTING.md](CONTRIBUTING.md).

No license has been selected for this repository. Public visibility alone does not grant an open-source license; clarify reuse terms with the owner before redistributing.
