import sysagent.auth as auth_mod
from sysagent.config import Settings

orig_find = auth_mod._find_agy_cli
try:
    auth_mod._find_agy_cli = lambda: "/usr/bin/agy"
    s = Settings(gemini_api_key="bad-key", gemini_api_keys=["bad-key"], auth_mode="api_key")
    mgr = auth_mod.AuthManager(s)
    mgr.initialize()
    assert mgr.status["mode"] == "api_key"
    assert mgr.rotate_key() is True
    assert mgr.status["mode"] == "agy_cli"
    assert isinstance(mgr.get_client(), auth_mod.AgyCliClient)
finally:
    auth_mod._find_agy_cli = orig_find
