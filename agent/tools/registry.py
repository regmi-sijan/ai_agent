import inspect
from typing import Any, Callable, Dict, List, Optional
import json

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}
        self._parameters: Dict[str, Dict[str, Any]] = {}

    def register(self, name: Optional[str] = None, description: Optional[str] = None):
        """Decorator to register a function as an agent tool."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or (func.__doc__ or "").strip()

            sig = inspect.signature(func)
            params: Dict[str, Any] = {
                "type": "OBJECT",
                "properties": {},
                "required": []
            }

            type_map = {
                str: "STRING",
                int: "INTEGER",
                float: "NUMBER",
                bool: "BOOLEAN",
                list: "ARRAY",
                dict: "OBJECT"
            }

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                param_type = type_map.get(param.annotation, "STRING")
                params["properties"][param_name] = {
                    "type": param_type,
                    "description": f"Parameter: {param_name}"
                }
                if param.default == inspect.Parameter.empty:
                    params["required"].append(param_name)

            self._tools[tool_name] = func
            self._descriptions[tool_name] = tool_desc
            self._parameters[tool_name] = params
            return func
        return decorator

    def get_tool(self, name: str) -> Optional[Callable]:
        return self._tools.get(name)

    def execute(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool by name with provided keyword arguments."""
        if name not in self._tools:
            return f"Error: Tool '{name}' not found."
        try:
            func = self._tools[name]
            result = func(**args)
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2)
            return str(result)
        except Exception as e:
            return f"Error executing '{name}': {str(e)}"

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return list of all registered tools with name and docstring."""
        return [
            {"name": name, "description": self._descriptions[name]}
            for name in self._tools
        ]

    def get_callable_list(self) -> List[Callable]:
        """Return raw list of python callables for Google GenAI automatic tool binding."""
        return list(self._tools.values())

# Global registry instance
registry = ToolRegistry()
register_tool = registry.register
