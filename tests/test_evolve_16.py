import asyncio
from unittest.mock import MagicMock
from sysagent.evolve import SelfEvolver
from sysagent.config import Settings

settings = Settings(evolve_enabled=True)
watcher = MagicMock()
watcher.current_context = 'normal'
storage = MagicMock()
storage.list_memories.return_value = []
storage.list_evolve_proposals.return_value = []
agent = MagicMock()
agent.auth.is_configured = True

evolver = SelfEvolver(settings, watcher, storage, agent)
evolver._last_codebase_hash = 'prev_hash'

async def fake_locked(*args, **kwargs):
    evolver._last_codebase_hash = 'new_hash'
    raise RuntimeError('transient quota error')

evolver._evolve_once_locked = fake_locked

async def main():
    try:
        await evolver.evolve_once(force=False)
        raise AssertionError('Expected RuntimeError')
    except RuntimeError:
        pass
    assert evolver._last_codebase_hash == 'prev_hash', f'Expected prev_hash, got {evolver._last_codebase_hash}'

asyncio.run(main())