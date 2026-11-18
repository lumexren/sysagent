import asyncio
from unittest.mock import MagicMock, patch
from sysagent.config import Settings
from sysagent.watcher import SystemWatcher

async def _test():
    storage = MagicMock()
    storage.list_automations.return_value = [{
        "id": 1,
        "name": "test_multi",
        "trigger_type": "multi_condition",
        "trigger_value": '{"max_cpu": 99}',
        "last_run_at": None,
    }]
    cb = MagicMock(return_value=asyncio.sleep(0))
    watcher = SystemWatcher(Settings(), storage=storage, automation_callback=cb)
    with patch("sysagent.watcher.asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
        await watcher._check_automations([])
        mock_to_thread.assert_called_once_with(watcher.get_health_snapshot)

asyncio.run(_test())