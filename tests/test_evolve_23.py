import inspect
from sysagent import brain

src = inspect.getsource(brain.CognitiveBrain.think_once)
assert "time.time_ns()" in src
assert "asyncio.to_thread(_prompt_dialog)" in src
