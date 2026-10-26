import os
import subprocess
from pathlib import Path
from agent.tools.registry import register_tool

@register_tool(name="run_shell_command", description="Run a safe shell command on the host system and return its output.")
def run_shell_command(command: str) -> str:
    """Execute a shell command with a timeout.

    Args:
        command: The shell command string to execute.
    """
    try:
        res = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = res.stdout
        if res.stderr:
            output += ("\n[stderr]\n" + res.stderr)
        return output.strip() if output.strip() else "(command completed with no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing shell command: {str(e)}"

@register_tool(name="read_file", description="Read the text content of a file at the given filepath.")
def read_file(filepath: str) -> str:
    """Read contents of a file.

    Args:
        filepath: Path to the file to read.
    """
    path = Path(filepath).expanduser().resolve()
    if not path.exists():
        return f"Error: File '{filepath}' does not exist."
    if not path.is_file():
        return f"Error: '{filepath}' is not a regular file."
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{filepath}': {str(e)}"

@register_tool(name="write_file", description="Write or overwrite text content to a file at the given filepath.")
def write_file(filepath: str, content: str) -> str:
    """Write text content to a file.

    Args:
        filepath: Path to the file to write.
        content: The text content to write.
    """
    path = Path(filepath).expanduser().resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {filepath}."
    except Exception as e:
        return f"Error writing file '{filepath}': {str(e)}"

@register_tool(name="list_directory", description="List files and folders in a specified directory.")
def list_directory(directory_path: str = ".") -> str:
    """List contents of a directory.

    Args:
        directory_path: The directory path to inspect (defaults to current directory).
    """
    path = Path(directory_path).expanduser().resolve()
    if not path.exists():
        return f"Error: Directory '{directory_path}' does not exist."
    if not path.is_dir():
        return f"Error: '{directory_path}' is not a directory."
    try:
        entries = sorted(os.listdir(path))
        items = []
        for entry in entries:
            full = path / entry
            tag = "[DIR]" if full.is_dir() else "[FILE]"
            items.append(f"{tag} {entry}")
        return "\n".join(items) if items else "(empty directory)"
    except Exception as e:
        return f"Error listing directory '{directory_path}': {str(e)}"

@register_tool(name="calculate", description="Evaluate a mathematical expression safely.")
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression string, e.g. '25 * 4 + 10' or 'sqrt(16)'.
    """
    import math
    safe_dict = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sum": sum,
        "math": math,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e
    }
    try:
        # Safe eval with restricted builtins
        res = eval(expression, {"__builtins__": {}}, safe_dict)
        return str(res)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"
