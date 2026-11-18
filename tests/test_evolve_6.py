import asyncio
from types import SimpleNamespace
from sysagent.auth import AuthManager
from sysagent.config import Settings

settings = Settings(gemini_api_key="", gemini_api_keys=["bad_key_1", "good_key_2"], auth_mode="api_key")
auth = AuthManager(settings)
auth.initialize()

calls = []

class DummyInteractions:
    def __init__(self, key):
        self.key = key
    def create(self, **kwargs):
        calls.append(self.key)
        if self.key == "bad_key_1":
            raise RuntimeError("403 PERMISSION_DENIED: Billing account not configured")
        return SimpleNamespace(output_text="ok")

orig_rotate = auth.rotate_key
auth._client = SimpleNamespace(interactions=DummyInteractions("bad_key_1"))
def mock_rotate():
    res = orig_rotate()
    auth._client = SimpleNamespace(interactions=DummyInteractions(auth._keys[auth._key_index]))
    return res
auth.rotate_key = mock_rotate

res = asyncio.run(auth.call_interactions(None, "gemini-3.5-flash-lite", input="hi"))
assert res.output_text == "ok"
assert calls == ["bad_key_1", "good_key_2"]