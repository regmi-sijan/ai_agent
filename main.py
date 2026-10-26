#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any

# Ensure current script directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

import config
from agent.core import Agent
from agent.tools import registry
from agent.subagents import AutonomousPhysicsOrchestrator

def print_banner(console):
    title = "[bold cyan]🔬 Autonomous Multi-Agent Physics & Scientific ML System[/bold cyan]"
    desc = (
        f"[green]Model:[/green] {config.DEFAULT_MODEL}  |  "
        "[magenta]Sub-Agent 1:[/magenta] Methodologist (Theory & Planning)  |  "
        "[cyan]Sub-Agent 2:[/cyan] Experimenter (Training & Physics Evaluation)\n"
        "[yellow]Commands:[/yellow] /pipeline <file> <desc>  |  /tools  |  /help  |  /exit"
    )
    console.print(Panel(f"{title}\n{desc}", border_style="cyan"))

def show_tools(console):
    table = Table(title="Available System Tools", border_style="dim")
    table.add_column("Tool Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    for tool in registry.list_tools():
        table.add_row(tool["name"], tool["description"] or "No description")
    console.print(table)

def run_autonomous_pipeline(console, dataset_path: str, description: str, target: str = "", pretrained_model: str = ""):
    """Run the 2-agent autonomous research and experiment pipeline."""
    if console:
        model_info = f"\n[white]Pre-trained Model:[/white] {pretrained_model}" if pretrained_model else ""
        console.print(Panel(
            f"[bold cyan]Starting Autonomous Dual-Agent Pipeline[/bold cyan]\n"
            f"[white]Dataset:[/white] {dataset_path}\n"
            f"[white]Description:[/white] {description}\n"
            f"[white]Target Criteria:[/white] {target or 'Autonomous Precision Optimization'}\n"
            f"[white]Live Steering:[/white] Write to [dim]outputs/checkpoints/live_directive.txt[/dim] to inject instructions mid-run"
            f"{model_info}",
            border_style="cyan"
        ))

    def on_subagent_start(agent_name: str, task: str):
        if console:
            icon = "🧠" if "Methodologist" in agent_name else "🧪"
            color = "magenta" if "Methodologist" in agent_name else "green"
            console.print(f"\n[bold {color}]{icon} [{agent_name} Sub-Agent][/bold {color}] {task}")
        else:
            print(f"\n>> [{agent_name}] {task}")

    def on_subagent_end(agent_name: str, output: str):
        if console:
            icon = "🧠" if "Methodologist" in agent_name else "🧪"
            color = "magenta" if "Methodologist" in agent_name else "green"
            title = f"[bold {color}]{icon} {agent_name} Report[/bold {color}]"
            console.print(Panel(Markdown(output), title=title, border_style=color))
        else:
            print(f"\n[{agent_name} Report]:\n{output}")

    def on_tool_call(agent_name: str, tool_name: str, args: Dict[str, Any]):
        if console:
            arg_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())
            console.print(f"   [yellow]⚙ [{agent_name} Tool][/yellow] [cyan]{tool_name}[/cyan]({arg_str})")
        else:
            print(f">> [{agent_name}] Tool: {tool_name}({args})")

    def on_tool_result(agent_name: str, tool_name: str, result: str):
        if console:
            truncated = result if len(result) < 250 else result[:250] + "... [truncated]"
            console.print(Panel(truncated, title=f"Output: {tool_name}", border_style="dim", expand=False))

    def on_crosstalk(sender: str, recipient: str, message: str):
        if console:
            icon = "🧠" if "Methodologist" in sender else "🧪"
            color = "magenta" if "Methodologist" in sender else "green"
            console.print(Panel(
                Markdown(message),
                title=f"[bold {color}]{icon} Cross-Talk: {sender} ➔ {recipient}[/bold {color}]",
                border_style=color
            ))
        else:
            print(f">> [Cross-Talk: {sender} -> {recipient}]:\n{message[:200]}...")

    orchestrator = AutonomousPhysicsOrchestrator(
        max_iterations=3,
        on_subagent_start=on_subagent_start,
        on_subagent_end=on_subagent_end,
        on_crosstalk=on_crosstalk,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result
    )

    try:
        report = orchestrator.run_pipeline(
            dataset_path=dataset_path,
            user_description=description,
            target_precision=target,
            pretrained_model_path=pretrained_model if pretrained_model else None
        )
        if console:
            status_color = "green" if "APPROVED" in str(report.get("status", "")) or report.get("status") == "SUCCESS" else "yellow"
            log_dir_info = f"\n📁 Archived Run Log & Codes: [cyan]{report.get('run_log_dir')}[/cyan]" if report.get("run_log_dir") else ""
            console.print(Panel(
                f"[bold {status_color}]Pipeline Finished with Status: {report['status']}[/bold {status_color}]\n"
                f"Iterations Completed: {report['iterations_completed']}\n"
                f"📁 Dedicated Run Folder: [cyan]{report.get('run_log_dir')}[/cyan]\n"
                f"   ├── figures/ (all plots & distributions)\n"
                f"   ├── models/ (all trained model checkpoints)\n"
                f"   ├── scripts/ (all executed Python/ROOT scripts)\n"
                f"   └── data/ (archived input datasets)",
                title="Pipeline Execution Complete",
                border_style=status_color
            ))
    except Exception as e:
        if console:
            console.print(f"[bold red]Pipeline Error:[/bold red] {e}")
        else:
            print(f"Pipeline Error: {e}")

def main():
    console = Console() if RICH_AVAILABLE else None

    parser = argparse.ArgumentParser(description="Autonomous Physics & Scientific ML Multi-Agent CLI")
    parser.add_argument("--gui", action="store_true", help="Launch the Web GUI dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Web GUI port (default: 8000)")
    parser.add_argument("--pipeline", action="store_true", help="Launch the dual-agent autonomous loop")
    parser.add_argument("--data", nargs="+", help="Path(s) to input dataset file(s) or folder (CSV, ROOT, NPY, etc.)")
    parser.add_argument("--model", type=str, default="", help="Path to pre-trained model file (.pt, .pth, .joblib) or Model ID")
    parser.add_argument("--task", type=str, help="Task description or physics goal")
    parser.add_argument("--target", type=str, default="", help="Precision criteria / target metric")
    parser.add_argument("query", nargs="*", help="Optional direct user query")

    args = parser.parse_args()

    # Launch GUI
    if args.gui:
        from gui.server import start_server
        start_server(port=args.port)
        return

    # Direct CLI pipeline trigger
    if args.pipeline:
        if not args.data or not args.task:
            print("Error: When using --pipeline, please specify --data <file(s)> and --task <description>.")
            sys.exit(1)
        data_arg = args.data[0] if len(args.data) == 1 else ", ".join(args.data)
        run_autonomous_pipeline(console, data_arg, args.task, args.target, args.model)
        return

    # Direct query mode
    if args.query:
        query_text = " ".join(args.query)
        agent = Agent(
            on_tool_call=lambda n, a: console.print(f"[bold yellow]⚙ Tool:[/bold yellow] [cyan]{n}[/cyan]({a})") if console else print(f">> Tool: {n}"),
            on_tool_result=lambda n, r: console.print(Panel(r[:300], title=n, border_style="dim")) if console else None
        )
        if console:
            console.print(f"[bold blue]User:[/bold blue] {query_text}")
            with console.status("[bold green]Executing...[/bold green]", spinner="dots"):
                ans = agent.run(query_text)
            console.print(Panel(Markdown(ans), title="[bold green]Response[/bold green]", border_style="green"))
        else:
            print(agent.run(query_text))
        return

    # Interactive REPL mode
    if console:
        print_banner(console)
    else:
        print("=== Autonomous Multi-Agent Physics CLI ===")

    # Initialize standard assistant agent for general queries
    agent = Agent(
        on_tool_call=lambda n, a: console.print(f"[bold yellow]⚙ Tool:[/bold yellow] [cyan]{n}[/cyan]({a})") if console else print(f">> Tool: {n}"),
        on_tool_result=lambda n, r: console.print(Panel(r[:300], title=n, border_style="dim")) if console else None
    )

    while True:
        try:
            if console:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
            else:
                user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                if console:
                    console.print("[dim]Goodbye![/dim]")
                break
            elif user_input.lower() == "/clear":
                agent.memory.clear()
                if console:
                    console.print("[yellow]Conversation history cleared.[/yellow]")
                continue
            elif user_input.lower() == "/tools":
                if console:
                    show_tools(console)
                continue
            elif user_input.lower() == "/help":
                help_text = (
                    "• /pipeline <file> <task description> : Launch the autonomous dual sub-agent loop\n"
                    "• /tools                                : List all available agent tools\n"
                    "• /clear                                : Reset conversation memory\n"
                    "• /exit                                 : Exit the agent CLI"
                )
                if console:
                    console.print(Panel(help_text, title="Available Commands", border_style="yellow"))
                else:
                    print(help_text)
                continue
            elif user_input.startswith("/pipeline"):
                parts = user_input.split(maxsplit=2)
                if len(parts) < 3:
                    if console:
                        console.print("[yellow]Usage: /pipeline <dataset_filepath> <task description>[/yellow]")
                    else:
                        print("Usage: /pipeline <dataset_filepath> <task description>")
                    continue
                data_path = parts[1]
                task_desc = parts[2]
                run_autonomous_pipeline(console, data_path, task_desc)
                continue

            # Standard chat interaction
            if console:
                with console.status("[bold green]Agents analyzing...[/bold green]", spinner="dots"):
                    answer = agent.run(user_input)
                console.print(Panel(Markdown(answer), title="[bold green]Response[/bold green]", border_style="green"))
            else:
                print("Thinking...")
                answer = agent.run(user_input)
                print(f"\n{answer}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            if console:
                console.print(f"[bold red]Error:[/bold red] {e}")
            else:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
