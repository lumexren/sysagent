from unittest.mock import patch
from sysagent.auth import AuthManager, AgyCliClient
from sysagent.config import Settings

s = Settings(gemini_api_key="", gemini_api_keys=[], auth_mode="adc")
mgr = AuthManager(s)
mgr._initialized = True
mgr._status = {"mode": "adc", "configured": True, "project": "broken-proj"}
with patch("sysagent.auth._find_agy_cli", return_value="/usr/bin/agy"):
    assert mgr.rotate_key() is True
    assert mgr.status["mode"] == "agy_cli"
    assert isinstance(mgr.get_client(), AgyCliClient)
