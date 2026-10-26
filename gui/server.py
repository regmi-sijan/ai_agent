import os
import sys
import json
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional
import threading

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from agent.subagents.orchestrator import AutonomousPhysicsOrchestrator

OUTPUT_DIR = BASE_DIR / "outputs"
RUNS_DIR = OUTPUT_DIR / "runs"
LOGS_DIR = OUTPUT_DIR / "logs"
FIGURES_DIR = OUTPUT_DIR / "figures"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
SCRIPTS_DIR = OUTPUT_DIR / "scripts"
MODELS_DIR = OUTPUT_DIR / "models"

for d in [OUTPUT_DIR, RUNS_DIR, LOGS_DIR, FIGURES_DIR, UPLOADS_DIR, SCRIPTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Antigravity Physics & Scientific ML Dashboard")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# Mount static figures
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

# Global event queues per session for SSE streaming
session_queues = {}
active_orchestrators = {}

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    template_path = Path(__file__).resolve().parent / "templates" / "index.html"
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))

@app.get("/api/figures")
async def list_figures():
    valid_exts = {".png", ".jpg", ".jpeg", ".svg"}
    figs = set()
    if FIGURES_DIR.exists():
        for f in FIGURES_DIR.iterdir():
            if f.suffix.lower() in valid_exts:
                figs.add(f.name)
    # Also discover and mirror figures from individual run folders
    if RUNS_DIR.exists():
        for r in RUNS_DIR.iterdir():
            fig_sub = r / "figures"
            if fig_sub.exists():
                for f in fig_sub.iterdir():
                    if f.suffix.lower() in valid_exts:
                        figs.add(f.name)
                        dest = FIGURES_DIR / f.name
                        if not dest.exists():
                            try:
                                import shutil
                                shutil.copy2(f, dest)
                            except Exception:
                                pass
    return JSONResponse(sorted(list(figs), reverse=True))

def resolve_dataset_path(raw_path: str) -> str:
    """Robustly resolve any user-provided path (quotes, tildes, relative paths, file or directory)."""
    if not raw_path:
        return str(UPLOADS_DIR)

    cleaned = raw_path.strip().strip("'\"").strip()
    if not cleaned:
        return str(UPLOADS_DIR)

    p = Path(cleaned).expanduser()

    # If absolute and exists
    if p.exists():
        return str(p.resolve())

    # Check relative to BASE_DIR (e.g. outputs/uploads/truth.csv or sample_physics_data.csv)
    if (BASE_DIR / p).exists():
        return str((BASE_DIR / p).resolve())

    # Check relative to OUTPUT_DIR
    if (OUTPUT_DIR / p).exists():
        return str((OUTPUT_DIR / p).resolve())

    # Check relative to UPLOADS_DIR
    if (UPLOADS_DIR / p).exists():
        return str((UPLOADS_DIR / p).resolve())

    # Check parent workspace
    if (BASE_DIR.parent / p).exists():
        return str((BASE_DIR.parent / p).resolve())

    # Return expanded/resolved path
    return str(p.resolve() if p.is_absolute() else (BASE_DIR / p).resolve())

@app.get("/api/check_path")
async def check_path(path: str = ""):
    """Real-time validation for user-provided dataset file or directory path."""
    if not path or not path.strip():
        return {"exists": False, "message": "No path provided"}

    resolved = resolve_dataset_path(path)
    p = Path(resolved)

    if not p.exists():
        return {
            "exists": False,
            "resolved_path": resolved,
            "message": f"Path not found: {p.name}"
        }

    if p.is_file():
        size_mb = round(p.stat().st_size / (1024 * 1024), 2)
        return {
            "exists": True,
            "type": "file",
            "filename": p.name,
            "size_mb": size_mb,
            "resolved_path": resolved,
            "message": f"Valid file: {p.name} ({size_mb} MB)"
        }

    if p.is_dir():
        valid_exts = {".csv", ".tsv", ".root", ".parquet", ".npy", ".npz", ".dat"}
        datasets = [f.name for f in p.glob("*") if f.suffix.lower() in valid_exts]
        return {
            "exists": True,
            "type": "directory",
            "datasets_found": len(datasets),
            "sample_files": datasets[:5],
            "resolved_path": resolved,
            "message": f"Valid folder: {len(datasets)} dataset(s) found"
        }

def run_pipeline_thread(session_id: str, dataset_path: str, task: str, target: str, pretrained_model_path: Optional[str] = None, existing_plan: Optional[str] = None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    queue = session_queues.get(session_id)

    def publish_event(event_dict: dict):
        if queue:
            queue.put_nowait(event_dict)

    def on_subagent_start(agent_name: str, message: str):
        publish_event({
            "type": "stage",
            "agent": agent_name,
            "message": message,
            "progress": 35 if "Methodologist" in agent_name else 65
        })

    def on_subagent_end(agent_name: str, content: str):
        if "Methodologist" in agent_name:
            publish_event({
                "type": "theorist_plan",
                "content": content
            })
        else:
            publish_event({
                "type": "experimenter_report",
                "content": content
            })

    def on_tool_call(agent_name: str, tool_name: str, args: dict):
        publish_event({
            "type": "tool_call",
            "agent": agent_name,
            "name": tool_name,
            "args": args
        })

    def on_tool_result(agent_name: str, tool_name: str, output: str):
        publish_event({
            "type": "tool_result",
            "agent": agent_name,
            "name": tool_name,
            "output": output
        })

    def on_crosstalk(sender: str, recipient: str, message: str):
        publish_event({
            "type": "crosstalk",
            "sender": sender,
            "recipient": recipient,
            "message": message
        })

    orchestrator = AutonomousPhysicsOrchestrator(
        max_iterations=3,
        run_id=None, # Automatically assigns sequential run_001, run_002, etc.
        on_subagent_start=on_subagent_start,
        on_subagent_end=on_subagent_end,
        on_crosstalk=on_crosstalk,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result
    )
    active_orchestrators[session_id] = orchestrator

    try:
        publish_event({
            "type": "stage",
            "message": f"Initializing {orchestrator.run_logger.run_id} and inspecting data...",
            "progress": 15
        })

        result = orchestrator.run_pipeline(
            dataset_path=dataset_path,
            user_description=task,
            target_precision=target,
            pretrained_model_path=pretrained_model_path,
            existing_plan=existing_plan
        )

        publish_event({
            "type": "pipeline_finished",
            "status": result.get("status", "SUCCESS"),
            "iterations": result.get("iterations_completed", 1),
            "run_id": orchestrator.run_logger.run_id,
            "run_log_dir": result.get("run_log_dir", "")
        })

    except Exception as e:
        run_log_dir = ""
        try:
            if orchestrator and orchestrator.run_logger:
                run_log_dir = str(orchestrator.run_logger.finalize_run(
                    dataset_path=dataset_path,
                    task=task,
                    target_precision=target,
                    status=f"ERROR: {str(e)}",
                    methodology_plan="",
                    final_result=f"Pipeline stopped with error: {str(e)}",
                    peer_review=None,
                    iterations=0
                ))
        except Exception:
            pass

        publish_event({
            "type": "pipeline_finished",
            "status": f"ERROR: {str(e)}",
            "iterations": 0,
            "run_log_dir": run_log_dir
        })
    finally:
        active_orchestrators.pop(session_id, None)

@app.post("/api/run")
async def run_analysis(
    task: str = Form(...),
    target: str = Form(""),
    mode: str = Form("upload"),
    folder_path: Optional[str] = Form(None),
    pretrained_model: Optional[str] = Form(None),
    existing_plan: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    model_file: Optional[UploadFile] = File(None)
):
    session_id = str(uuid.uuid4())
    queue = asyncio.Queue()
    session_queues[session_id] = queue

    target_path = ""
    enriched_task = task
    valid_files = [f for f in files if f and hasattr(f, "filename") and f.filename and f.filename.strip()] if files else []

    if mode == "upload" and valid_files:
        if len(valid_files) > 1:
            run_dir = UPLOADS_DIR / f"run_{session_id[:8]}"
            run_dir.mkdir(parents=True, exist_ok=True)
            saved_paths = []
            for file in valid_files:
                fname = Path(file.filename).name
                dest = run_dir / fname
                content = await file.read()
                with open(dest, "wb") as f:
                    f.write(content)
                saved_paths.append(str(dest))
            target_path = str(run_dir)
            filenames_str = ", ".join(Path(f.filename).name for f in valid_files)
            enriched_task += f"\n[UPLOADED DATASETS]: {len(valid_files)} files uploaded to directory '{target_path}': [{filenames_str}]. Inspect the directory using `inspect_data_folder` or each file using `inspect_numerical_dataset`."
        else:
            fname = Path(valid_files[0].filename).name
            dest = UPLOADS_DIR / fname
            content = await valid_files[0].read()
            with open(dest, "wb") as f:
                f.write(content)
            target_path = str(dest)
    else:
        target_path = resolve_dataset_path(folder_path)

    model_dest_path = None
    if model_file and hasattr(model_file, "filename") and model_file.filename and model_file.filename.strip():
        fname_model = Path(model_file.filename).name
        dest_model = UPLOADS_DIR / fname_model
        content = await model_file.read()
        with open(dest_model, "wb") as f:
            f.write(content)
        model_dest_path = str(dest_model)
    elif pretrained_model and pretrained_model.strip():
        model_dest_path = resolve_dataset_path(pretrained_model.strip())

    # Start autonomous orchestrator in background thread
    t = threading.Thread(
        target=run_pipeline_thread,
        args=(session_id, target_path, enriched_task, target, model_dest_path, existing_plan),
        daemon=True
    )
    t.start()

    return {"session_id": session_id, "dataset_path": target_path, "model_path": model_dest_path}

@app.get("/api/stream/{session_id}")
async def event_stream(session_id: str):
    queue = session_queues.get(session_id)
    if not queue:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    async def sse_generator():
        while True:
            try:
                # Wait for next event from orchestrator
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "pipeline_finished":
                    break
            except asyncio.CancelledError:
                break

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.post("/api/steer")
async def steer_pipeline(
    session_id: str = Form(...),
    instruction: str = Form(...)
):
    """Mid-run steering directive injected by the user while the pipeline is processing."""
    orchestrator = active_orchestrators.get(session_id)
    if not orchestrator:
        return JSONResponse({"error": "No active pipeline currently running for this session."}, status_code=404)

    clean_instruction = instruction.strip()
    if not clean_instruction:
        return JSONResponse({"error": "Instruction text cannot be empty."}, status_code=400)

    # Inject into the orchestrator queue
    orchestrator.inject_user_instruction(clean_instruction)

    # Notify GUI stream immediately
    queue = session_queues.get(session_id)
    if queue:
        queue.put_nowait({
            "type": "crosstalk",
            "sender": "User Steering (Live)",
            "recipient": "All Sub-Agents",
            "message": f"⚡ **Live Directive Injected:** {clean_instruction}"
        })
        queue.put_nowait({
            "type": "stage",
            "agent": "Orchestrator",
            "message": f"⚡ Live User Directive Injected: '{clean_instruction}'",
            "progress": 55
        })

    return {"status": "ok", "message": "Directive delivered to active sub-agents."}

@app.get("/api/logs")
async def list_runs():
    """List archived run folders."""
    discovered = {}
    candidate_dirs = [RUNS_DIR, LOGS_DIR]

    for c_dir in candidate_dirs:
        if c_dir.exists():
            for d in c_dir.iterdir():
                if d.is_dir() and d.name.startswith("run_") and d.name not in discovered:
                    summary_file = d / "summary.md"
                    results_file = d / "results.json"
                    meta = {}
                    if results_file.exists():
                        try:
                            meta = json.loads(results_file.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                    discovered[d.name] = {
                        "run_id": d.name,
                        "path": str(d),
                        "timestamp": meta.get("timestamp", d.name.replace("run_", "")),
                        "status": meta.get("status", "UNKNOWN"),
                        "task": meta.get("task", ""),
                        "has_summary": summary_file.exists()
                    }

    # Sort runs so highest/latest run number is first (e.g. run_002, run_001)
    sorted_runs = sorted(discovered.values(), key=lambda r: r["run_id"], reverse=True)
    return JSONResponse(sorted_runs)

@app.get("/api/logs/{run_id}")
async def get_run_details(run_id: str):
    """Retrieve full summary and results for a specific run."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        run_dir = LOGS_DIR / run_id
    if not run_dir.exists():
        return JSONResponse({"error": "Run not found"}, status_code=404)

    summary_file = run_dir / "summary.md"
    results_file = run_dir / "results.json"
    dialogue_file = run_dir / "inter_agent_dialogue.md"

    return JSONResponse({
        "run_id": run_id,
        "summary": summary_file.read_text(encoding="utf-8") if summary_file.exists() else "",
        "dialogue": dialogue_file.read_text(encoding="utf-8") if dialogue_file.exists() else "",
        "results": json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else {}
    })


def start_server(host: str = "127.0.0.1", port: int = 8000, auto_open: bool = True):
    import uvicorn
    import webbrowser
    import time

    url = f"http://{host}:{port}"
    print(f"\n🚀 Launching Antigravity Physics & Scientific ML Dashboard on {url}\n")

    if auto_open:
        def open_browser():
            time.sleep(1.2)
            print(f"🌐 Automatically opening browser at {url} ...")
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
