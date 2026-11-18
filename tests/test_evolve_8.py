import asyncio
from unittest.mock import MagicMock
from sysagent.agent import AgentManager, PendingApproval
from sysagent.config import Settings

settings = Settings()
storage = MagicMock()
watcher = MagicMock()
agent = AgentManager(settings, watcher, storage)

loop = asyncio.new_event_loop()
fut = loop.create_future()
agent.pending_approvals["ap1"] = PendingApproval(
    approval_id="ap1",
    session_id="s1",
    tool_name="run_command",
    arguments={"cmd": "ls"},
    reason="test",
    risk_level="medium",
    future=fut,
)
agent.reset_session("s1")
assert "ap1" not in agent.pending_approvals
assert fut.done() and fut.result() is False
loop.close()