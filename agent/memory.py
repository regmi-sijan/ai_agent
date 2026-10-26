from typing import List, Dict, Any

class ConversationMemory:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "parts": [content]})

    def add_model_message(self, content: str):
        self.history.append({"role": "model", "parts": [content]})

    def clear(self):
        self.history.clear()

    def get_messages(self) -> List[Dict[str, Any]]:
        return list(self.history)
