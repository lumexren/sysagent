from unittest.mock import patch
import httpx
from sysagent.cli import _cmd_status

with patch("sysagent.cli.httpx.get", side_effect=httpx.ConnectError("offline")), patch("sysagent.cli.Storage"):
    try:
        _cmd_status("http://127.0.0.1:9999", locale="en")
        raise AssertionError("Expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 1