import time
import random
from typing import Callable, Optional, Any, List
import config

_LAST_CALL_TIMESTAMP = 0.0

def _apply_rate_limit_throttle():
    """Proactive RPM rate-limiter: ensure at least RATE_LIMIT_DELAY between requests."""
    global _LAST_CALL_TIMESTAMP
    min_interval = getattr(config, "RATE_LIMIT_DELAY", 2.5)
    now = time.time()
    elapsed = now - _LAST_CALL_TIMESTAMP
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_CALL_TIMESTAMP = time.time()


class ResilientChatSession:
    """Manages a Gemini chat session with automatic multi-architecture failover.
    When a model hits a usage limit (429 RESOURCE_EXHAUSTED / quota limit),
    it automatically migrates conversation history to the next efficient architecture in the cascade.
    """

    def __init__(
        self,
        client: Any,
        system_instruction: str,
        tools: Optional[List[Callable]] = None,
        temperature: float = 0.2,
        on_log: Optional[Callable[[str], None]] = None,
        models: Optional[List[str]] = None,
    ):
        self.client = client
        self.system_instruction = system_instruction
        self.tools = tools or []
        self.temperature = temperature
        self.on_log = on_log
        self.models = list(models or config.EFFICIENT_MODEL_ARCHITECTURES)
        self.current_model_idx = 0
        self.chat = None
        self._init_chat()

    def _init_chat(self, history: Optional[List[Any]] = None):
        from google.genai import types
        model_name = self.models[self.current_model_idx]
        gen_config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=self.tools if self.tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ) if self.tools else None,
            temperature=self.temperature,
        )
        kwargs = {"model": model_name, "config": gen_config}
        if history:
            kwargs["history"] = history
        self.chat = self.client.chats.create(**kwargs)

    @property
    def current_model(self) -> str:
        return self.models[self.current_model_idx]

    def get_history(self) -> List[Any]:
        if self.chat:
            try:
                return self.chat.get_history()
            except Exception:
                return []
        return []

    def send_message(
        self,
        message: Any,
        max_transient_retries: int = 4,
        initial_wait: float = 3.0
    ) -> Any:
        """Send a message to the active model architecture with automatic multi-model failover on 503 or 429."""
        while self.current_model_idx < len(self.models):
            current_model = self.models[self.current_model_idx]

            for attempt in range(max_transient_retries):
                _apply_rate_limit_throttle()
                try:
                    return self.chat.send_message(message)
                except Exception as e:
                    err_str = str(e).lower()
                    is_quota_exhausted = (
                        "429" in err_str or
                        "resource_exhausted" in err_str or
                        "quota" in err_str or
                        "limit exceeded" in err_str
                    )
                    is_transient_surge = (
                        "503" in err_str or
                        "unavailable" in err_str or
                        "high demand" in err_str or
                        "overloaded" in err_str or
                        "connection" in err_str
                    )
                    is_model_unavailable = (
                        "404" in err_str or
                        "not_found" in err_str or
                        "no longer available" in err_str or
                        "deprecated" in err_str or
                        "not found for api version" in err_str
                    )

                    # Automatically fail over to the next available model in the cascade
                    if is_quota_exhausted or is_transient_surge or is_model_unavailable:
                        if is_transient_surge:
                            reason = "high demand / server surge (503)"
                        elif is_quota_exhausted:
                            reason = "usage limit (429)"
                        else:
                            reason = "model deprecation / unavailable (404)"

                        if self.current_model_idx + 1 < len(self.models):
                            old_model = current_model
                            self.current_model_idx += 1
                            new_model = self.models[self.current_model_idx]

                            failover_msg = (
                                f"🔄 [Automatic Model Failover] Model '{old_model}' encountered {reason}. "
                                f"Automatically switching to '{new_model}' and continuing seamlessly..."
                            )
                            print(f"\n{failover_msg}")
                            if self.on_log:
                                self.on_log(failover_msg)

                            # Transfer conversation history to the new model
                            history = self.get_history()
                            self._init_chat(history=history)
                            break  # Exit inner retry loop to try the new model immediately
                        else:
                            # Reached the end of available models; retry with backoff on the final model
                            if attempt < max_transient_retries - 1:
                                wait_time = initial_wait * (2 ** attempt) + random.uniform(1.0, 2.0)
                                surge_msg = f"⚠️ All alternative models tried. Retrying on '{current_model}' in {round(wait_time, 1)}s..."
                                print(f"\n{surge_msg}")
                                if self.on_log:
                                    self.on_log(surge_msg)
                                time.sleep(wait_time)
                            else:
                                raise e
                    else:
                        raise e

        raise RuntimeError(f"All {len(self.models)} model architectures exhausted their limits.")


def send_message_with_retry(
    chat: Any,
    message: Any,
    max_retries: int = 5,
    initial_wait: float = 4.0,
    on_retry: Optional[Callable[[int, float, str], None]] = None
) -> Any:
    """Send a message to Google Gemini with automatic rate limiting and exponential backoff."""
    _apply_rate_limit_throttle()

    for attempt in range(max_retries):
        try:
            return chat.send_message(message)
        except Exception as e:
            err_str = str(e).lower()
            is_transient = (
                "503" in err_str or
                "unavailable" in err_str or
                "high demand" in err_str or
                "429" in err_str or
                "resource_exhausted" in err_str or
                "overloaded" in err_str or
                "connection" in err_str
            )
            if is_transient and attempt < max_retries - 1:
                base_wait = 6.0 if "429" in err_str or "resource_exhausted" in err_str else initial_wait
                wait_time = base_wait * (2 ** attempt) + random.uniform(0.5, 2.0)
                if on_retry:
                    on_retry(attempt + 1, round(wait_time, 1), str(e))
                time.sleep(wait_time)
            else:
                raise e
