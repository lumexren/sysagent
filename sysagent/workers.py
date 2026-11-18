"""High-efficiency parallel sub-agent worker pool for SysAgent."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from google import genai

from .auth import AuthManager
from .config import Settings
from .prompts import PromptRegistry
from .storage import Storage
from .tools import TOOL_DECLARATIONS, execute_tool
from .watcher import SystemWatcher


WORKER_ALLOWED_TOOLS = frozenset(
    {
        "read_file",
        "list_directory",
        "search_files",
        "system_status",
        "list_processes",
        "list_systemd_units",
        "read_journal_logs",
        "kg_query",
        "recall_memories",
        "search_memory_and_history",
    }
)

WORKER_TOOLS = [d for d in TOOL_DECLARATIONS if d["name"] in WORKER_ALLOWED_TOOLS]


class WorkerPool:
    """Runs bounded-concurrency, stateless, read-only sub-agents in parallel."""

    def __init__(
        self,
        settings: Settings,
        watcher: SystemWatcher,
        storage: Storage,
        auth: AuthManager | None = None,
    ) -> None:
        self.settings = settings
        self.watcher = watcher
        self.storage = storage
        self.auth = auth or AuthManager(settings)

    def _get_client(self) -> genai.Client:
        return self.auth.get_client()

    async def _run_one(
        self, sem: asyncio.Semaphore, idx: int, task: str, max_steps: int = 6
    ) -> dict[str, Any]:
        async with sem:
            sys_inst = PromptRegistry(self.storage).render("worker_slice")
            worker_tools = WORKER_TOOLS
            try:
                from .neural import NeuralEngine

                eng = NeuralEngine(self.storage)
                if int(eng.meta.get("epochs_trained", 0)) > 0:
                    probs = eng.predict_tool_probs(task)
                    keep = {"read_file", "list_directory", "search_files", "system_status"}
                    for tname, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:4]:
                        if p >= 0.08 and tname in WORKER_ALLOWED_TOOLS:
                            keep.add(tname)
                    filtered = [t for t in WORKER_TOOLS if t.get("name") in keep]
                    if filtered:
                        worker_tools = filtered
            except Exception:
                pass
            next_input: Any = task
            prev_id: str | None = None
            for _ in range(max_steps):
                kwargs: dict[str, Any] = {
                    "model": self.settings.worker_model,
                    "input": next_input,
                    "tools": worker_tools,
                    "system_instruction": sys_inst,
                }
                if prev_id:
                    kwargs["previous_interaction_id"] = prev_id
                interaction = await self.auth.call_interactions(
                    self.storage, timeout_sec=60.0, **kwargs
                )
                prev_id = interaction.id
                calls = [
                    s for s in (interaction.steps or []) if s.type == "function_call"
                ]
                if not calls:
                    return {
                        "worker": idx,
                        "task": task,
                        "status": "ok",
                        "result": interaction.output_text or "",
                    }

                async def _exec_call(c: Any) -> dict[str, Any]:
                    args = dict(c.arguments or {})
                    if c.name not in WORKER_ALLOWED_TOOLS:
                        res: Any = json.dumps({"error": f"Tool not allowed in worker: {c.name}"})
                    else:
                        res = await asyncio.to_thread(
                            execute_tool, c.name, args, self.watcher, self.storage
                        )
                    return {
                        "type": "function_result",
                        "call_id": c.id,
                        "name": c.name,
                        "result": res if isinstance(res, str) else json.dumps(res),
                    }

                results = list(await asyncio.gather(*(_exec_call(c) for c in calls)))
                next_input = results

            return {
                "worker": idx,
                "task": task,
                "status": "max_steps_reached",
                "result": "Worker reached step limit before finalizing.",
            }

    async def run_parallel(self, tasks: list[str]) -> list[dict[str, Any]]:
        """Execute a list of subtasks concurrently with bounded semaphore."""
        if not tasks:
            return []
        sem = asyncio.Semaphore(max(1, self.settings.worker_max_concurrency))
        coros = [self._run_one(sem, i + 1, t) for i, t in enumerate(tasks)]
        raw = await asyncio.gather(*coros, return_exceptions=True)
        out: list[dict[str, Any]] = []
        for i, item in enumerate(raw):
            if isinstance(item, Exception):
                out.append(
                    {"worker": i + 1, "task": tasks[i], "status": "error", "result": str(item)}
                )
            else:
                out.append(item)
        return out
