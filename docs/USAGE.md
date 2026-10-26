# Usage and analysis prompts

Start with dataset inspection. Specify units, branch/column names, labels, missing-value conventions, and the scientific assumptions you want checked. Ask for baseline comparisons, held-out evaluation, uncertainty estimates, and executable artifacts. Inspect the generated code before relying on its conclusions.

## Included sample

`sample_physics_data.csv` contains ten rows with `event_id`, `px`, `py`, `pz`, `E`, and `particle_label`. Units and physical validity are not established. Use it for loading and plotting demonstrations, not for measured performance or mass reconstruction claims.

An offline inspection needs no API key:

```bash
python -c "from agent.tools.physics_tools import inspect_numerical_dataset; print(inspect_numerical_dataset('sample_physics_data.csv'))"
```

## Example scientific requests

**Kinematics:** “Inspect the collision data. Treat momenta and energies as GeV with c=1. Check whether E²−|p|² is nonnegative within numerical tolerance, flag inconsistent rows, and plot invariant mass and transverse momentum. Explain all selections and uncertainties.”

**Spectrum fitting:** “Inspect the energy and counts columns. Fit a Gaussian peak plus a justified background, report centroid and FWHM with uncertainties, and plot residuals. Compare alternative backgrounds before interpreting the fit.”

**Classification:** “Separate signal and background using the supplied label. Exclude event identifiers and target leakage. Split train/validation/test before scaling. Compare logistic regression or a tree baseline to a neural network and report held-out ROC AUC with uncertainties.”

**Waveforms:** “Inspect waveform shapes and sample spacing. Establish a baseline and pulse integration windows, calculate tail/total charge, and compare a simple discrimination threshold with a classifier. Explain how noisy and negative-charge pulses are handled.”

These are requests to the model, not built-in verified pipelines.

## File preparation

- CSV/TSV/text: include headers; numeric statistics are based on at most the first 1,000 rows. Estimated total rows come from line counting and may be inaccurate for multiline CSV records.
- ROOT: inspection lists keys, tree branches, and a small sample using uproot. Complex nested branches may require custom code.
- NPY: numeric arrays; object arrays requiring pickle are unsupported by the inspector.
- NPZ: named array archive; inspection reports member shapes and dtypes.
- Parquet: requires PyArrow; the current reader loads the file rather than streaming it.
- Model files: local `.pt`, `.pth`, `.bin`, `.joblib`, or `.pkl` checkpoints from trusted sources. A typed model ID is passed as context; automatic model downloads are not a guaranteed feature.

The inspector can load large NumPy and Parquet inputs into memory. Start with a small representative file. Directory discovery is recursive, but archived directory copies only include eligible top-level files. Duplicate basenames can collide in archives or the shared figure gallery.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Missing Gemini key | Copy `.env.example` to `.env` and supply `GEMINI_API_KEY`, or export it in the shell. |
| Model not found / 404 | Choose a tool-calling model available to your account; update `AGENT_MODEL` and optional fallback IDs. |
| Quota / 429 / overloaded / 503 | Check provider quota and billing, reduce run scope, and increase `RATE_LIMIT_DELAY`. Fallbacks also consume usage. |
| `root` not found | Install CERN ROOT separately or request a Python/uproot approach. |
| Missing module in generated code | Review the script and install its additional dependency into the active environment. |
| Port already used | Run `python main.py --gui --port 8080`. |
| Empty figures tab | Check script output and whether plots were saved to `os.environ['FIGURES_DIR']`. |
| Target not achieved | Review scripts, datasets, validation design, and model critique. A completed run is not an accuracy guarantee. |
| Dashboard styles/icons missing | Check connectivity to the CDNs used by the HTML template. |
| Training exceeds five minutes | The execution tool times out at 300 seconds; reduce the task or review and run the generated script separately. |

To reuse a plan through the CLI, include “resume” in the task only when `outputs/checkpoints/latest_plan.md` belongs to the same dataset and objective. The GUI can submit its existing plan directly. Neither path resumes a training process.
