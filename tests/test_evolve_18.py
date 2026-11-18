import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from sysagent.server import TerminalCommand, api_terminal_run

async def _test():
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_shell", AsyncMock(return_value=mock_proc)):
        res = await api_terminal_run(TerminalCommand(command="sleep 60"))

    assert res["returncode"] == -1
    assert "timed out" in res["stderr"]
    mock_proc.kill.assert_called_once()
    mock_proc.wait.assert_awaited_once()

asyncio.run(_test())