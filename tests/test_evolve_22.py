import asyncio
from types import SimpleNamespace
from sysagent.agent import AgentManager

class DummyInteractions:
    def __init__(self):
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("previous_interaction_id"):
            raise RuntimeError("400 invalid_request: invalid previous_interaction_id")
        if isinstance(kwargs.get("input"), list):
            raise RuntimeError("400 invalid_request: function_result requires previous_interaction_id")
        return SimpleNamespace(id="new-id", output_text="ok", steps=[])

class DummyClient:
    def __init__(self):
        self.interactions = DummyInteractions()

client = DummyClient()
res = asyncio.run(AgentManager._create_safe(client, {
    "model": "gemini-2.5-flash",
    "previous_interaction_id": "stale-chain-id",
    "input": [{"type": "function_result", "call_id": "c1", "name": "system_status", "result": "{\"cpu\": 10}"}],
}))
assert res.id == "new-id"
assert len(client.interactions.calls) == 2
assert "previous_interaction_id" not in client.interactions.calls[1]
assert isinstance(client.interactions.calls[1]["input"], str)
assert "system_status" in client.interactions.calls[1]["input"]
