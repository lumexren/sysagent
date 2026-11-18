from unittest.mock import patch
from sysagent.auth import AuthManager
from sysagent.config import Settings

s = Settings(gemini_api_key="test-key-1")
am = AuthManager(s)
am._initialized = True
am._keys = ["test-key-1"]
am._status = {"mode": "agy_cli", "configured": True, "project": "google-ai-pro"}
with patch("sysagent.auth.genai.Client") as mock_client:
    assert am.rotate_key() is True
    assert am._status["mode"] == "api_key"
    mock_client.assert_called_once_with(api_key="test-key-1")