# Architecture

`main.py` selects a single-query assistant, REPL, headless pipeline, or local GUI. Importing the tool package registers Python callables and creates output directories.

## Pipeline lifecycle

1. `AutonomousPhysicsOrchestrator` allocates a numbered `RunLogger` directory and archives eligible dataset files.
2. `MethodologistAgent` inspects inputs and produces a plan, or the orchestrator accepts a previous plan.
3. `ExperimenterAgent` receives the plan, writes and executes scripts, and may consult the Methodologist.
4. The Methodologist reviews the experiment's text report. A recognized `DECISION: APPROVED` marker ends the run; otherwise the plan is revised until the three-iteration cap.
5. The logger writes the summary, plan, dialogue, metadata, and available artifacts.

The model's approval marker controls status; it is not independently calculated from numeric target thresholds. There is no persistent job scheduler or distributed execution system.

## Tools and model sessions

The registry in `agent/tools/registry.py` exposes filesystem, shell, arithmetic, dataset inspection, checkpoint inspection, search, scientific execution, and consultation tools. Pipeline agents each use an allowlist. `ResilientChatSession` uses the configured primary and optional fallback model IDs, carrying chat history across a fallback where possible. The general assistant uses a single selected model with retries.

SDK automatic function execution is disabled so the application's own loops execute and record tools. Round limits bound these loops; they are not global limits on nested consultations or provider token usage.

Generated Python scripts run through the current Python interpreter from the project directory with Matplotlib's noninteractive backend. They receive `FIGURES_DIR`, `MODELS_DIR`, and `SCRIPTS_DIR` environment variables. C++ macros invoke a separately installed `root` command. No process isolation or permission boundary is added.

## Dashboard

FastAPI serves `gui/templates/index.html`, accepts uploads and run requests, starts a worker thread, and forwards events through server-sent events. Routes include `/api/run`, `/api/stream/{session_id}`, `/api/steer`, `/api/figures`, `/api/check_path`, and `/api/logs`.

Session queues live in process memory. Outputs, active-run context, and consultation routing include shared globals. Use one local run at a time; this is not a production multi-user server. Browser rendering depends on external CDNs. A restart loses live event/session state but preserves files.

## Extension points

- Register tools with `@register_tool` and add them to the appropriate agent allowlist.
- Adjust prompts in the Methodologist and Experimenter modules.
- Set `max_iterations` when constructing an orchestrator for programmatic use.
- Add real numeric validation before treating sanity-check reports as evidence.
- For production work, first add isolation, authenticated access, bounded uploads, safe rendering, and per-run state management.
