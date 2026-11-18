"""FastAPI daemon server for SysAgent (WebSocket + REST API)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .agent import AgentManager
from .auth import AuthManager
from .brain import CognitiveBrain
from .config import get_settings, persist_env_var
from .evolve import SelfEvolver
from .i18n import get_translations, resolve_locale, t
from .plugins import PluginRegistry
from .prompts import PromptRegistry
from .self_ops import SelfMonitor, export_brain_bundle, import_brain_bundle
from .self_learn import evolve_status_summary, learn_status_summary

_BOOT_TIME = time.monotonic()
from .storage import Storage
from .watcher import SystemWatcher

settings = get_settings()
auth = AuthManager(settings)
storage = Storage()
plugins = PluginRegistry()
watcher = SystemWatcher(settings, storage=storage)
agent = AgentManager(settings, watcher, storage, plugins=plugins, auth=auth)
brain = CognitiveBrain(settings, watcher, storage, agent)
evolver = SelfEvolver(settings, watcher, storage, agent)
self_monitor = SelfMonitor(storage, watcher, plugins=plugins, agent=agent, evolver=evolver)


async def _run_automation(auto: dict[str, Any], trigger_context: str) -> None:
    prompt = f"[Automation '{auto['name']}'] Trigger: {trigger_context}\nTask: {auto['prompt']}"
    run_sid = f"auto-{auto['id']}-{uuid.uuid4().hex[:6]}"
    try:
        reply = await agent.run_turn(run_sid, prompt)
    finally:
        agent.reset_session(run_sid)
    watcher._send_desktop_notification(f"{settings.agent_name}: {auto['name']}", reply[:200])


async def _inspect_alert(alert_message: str) -> None:
    prompt = (
        f"A background system alert was triggered and the user clicked 'Inspect':\n{alert_message}\n"
        "Inspect the relevant system state using read-only tools and summarize what is happening and what to do."
    )
    run_sid = f"inspect-{uuid.uuid4().hex[:6]}"
    try:
        reply = await agent.run_turn(run_sid, prompt)
    finally:
        agent.reset_session(run_sid)
    watcher._send_desktop_notification(f"{settings.agent_name} Analysis", reply[:240])


watcher.automation_callback = _run_automation
watcher.inspect_callback = _inspect_alert


@asynccontextmanager
async def lifespan(_app: FastAPI):
    auth.initialize()
    watcher.start()
    brain.start()
    evolver.start()
    self_monitor.start()
    try:
        yield
    finally:
        await self_monitor.stop()
        await evolver.stop()
        await brain.stop()
        await watcher.stop()


app = FastAPI(title="SysAgent Daemon", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if settings.auth_token and request.url.path.startswith("/api/"):
        auth_hdr = request.headers.get("authorization", "")
        query_tok = request.query_params.get("token", "")
        bearer = auth_hdr[7:].strip() if auth_hdr.lower().startswith("bearer ") else ""
        if bearer != settings.auth_token and query_tok != settings.auth_token:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


class ApprovalDecision(BaseModel):
    approved: bool


class LocaleUpdate(BaseModel):
    locale: str


class NameUpdate(BaseModel):
    name: str


class TerminalCommand(BaseModel):
    command: str


@app.get("/")
async def index() -> dict[str, str]:
    return {"service": settings.agent_name, "status": "running"}


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    a_status = auth.status
    return {
        "status": "running",
        "agent_name": settings.agent_name,
        "model": settings.model,
        "models": {
            "chat": settings.model,
            "light": settings.light_model,
            "evolve": settings.evolve_model,
            "worker": settings.worker_model,
        },
        "approval_mode": settings.approval_mode,
        "locale": settings.locale,
        "api_key_configured": a_status["configured"],
        "auth_mode": a_status["mode"],
        "health": watcher.get_health_snapshot(),
        "workspace_context": watcher.current_context,
        "quota_today": storage.get_model_quota_today(),
        "pending_approvals": len(agent.pending_approvals),
        "pending_brain_questions": len(storage.list_brain_questions("pending")),
        "brain_rules": len(storage.list_brain_rules()),
        "evolve": evolve_status_summary(storage, evolver, settings),
        "learn": learn_status_summary(storage),
        "evolve_proposals": len(storage.list_evolve_proposals(limit=50)),
        "memories": len(storage.list_memories()),
        "lessons": len(storage.list_lessons()),
        "uptime_sec": int(time.monotonic() - _BOOT_TIME),
    }


@app.get("/api/i18n")
async def api_i18n(locale: str | None = None) -> dict[str, Any]:
    target = locale or settings.locale
    return {"locale": resolve_locale(target), "strings": get_translations(target)}


@app.post("/api/locale")
async def api_set_locale(body: LocaleUpdate) -> dict[str, str]:
    settings.locale = body.locale.strip() or "auto"
    persist_env_var("SYSAGENT_LOCALE", settings.locale)
    return {"status": "updated", "locale": settings.locale}


@app.post("/api/name")
async def api_set_name(body: NameUpdate) -> dict[str, str]:
    new_name = body.name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Agent name cannot be empty")
    settings.agent_name = new_name
    persist_env_var("SYSAGENT_NAME", new_name)
    return {"status": "updated", "agent_name": settings.agent_name}


@app.post("/api/auth/reload")
async def api_auth_reload() -> dict[str, Any]:
    res = auth.reload()
    for field_name in vars(auth.settings):
        setattr(settings, field_name, getattr(auth.settings, field_name))
    return res


@app.get("/api/events")
async def api_events(limit: int = 30) -> dict[str, Any]:
    return {"events": watcher.get_events(limit=limit)}


@app.post("/api/approve/{approval_id}")
async def api_approve(approval_id: str, body: ApprovalDecision) -> dict[str, Any]:
    ok = agent.resolve_approval(approval_id, body.approved)
    if not ok:
        raise HTTPException(status_code=404, detail="Approval request not found or expired")
    return {"status": "resolved", "approved": body.approved}


@app.post("/api/reset/{session_id}")
async def api_reset(session_id: str) -> dict[str, str]:
    agent.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/api/sessions")
async def api_sessions() -> dict[str, Any]:
    return {"sessions": storage.list_sessions()}


@app.get("/api/sessions/{session_id}/messages")
async def api_session_messages(session_id: str) -> dict[str, Any]:
    return {"messages": storage.get_messages(session_id)}


@app.get("/api/memories")
async def api_memories() -> dict[str, Any]:
    return {"memories": storage.list_memories()}


@app.delete("/api/memories/{key}")
async def api_delete_memory(key: str) -> dict[str, Any]:
    return {"deleted": storage.delete_memory(key)}


@app.get("/api/automations")
async def api_automations() -> dict[str, Any]:
    return {"automations": storage.list_automations()}


@app.delete("/api/automations/{auto_id}")
async def api_delete_automation(auto_id: int) -> dict[str, Any]:
    return {"deleted": storage.delete_automation(auto_id)}


@app.get("/api/brain")
async def api_brain() -> dict[str, Any]:
    return {
        "enabled": settings.brain_enabled,
        "confidence_threshold": settings.brain_confidence_threshold,
        "quiet_hours": settings.quiet_hours,
        "quiet_now": brain.is_quiet_hours(),
        "questions": storage.list_brain_questions("pending"),
        "rules": storage.list_brain_rules(),
        "thoughts": storage.list_brain_thoughts(limit=20),
    }


@app.post("/api/brain/think")
async def api_brain_think() -> dict[str, Any]:
    return await brain.think_once(force=True)


@app.get("/api/brain/digest")
async def api_brain_digest() -> dict[str, Any]:
    return await brain.generate_daily_digest()


@app.get("/api/brain/questions")
async def api_brain_questions(status: str = "pending") -> dict[str, Any]:
    return {"questions": storage.list_brain_questions(status)}


@app.post("/api/brain/answer/{qid}")
async def api_brain_answer(qid: int, answer_body: dict[str, Any]) -> dict[str, Any]:
    answer_text = str(answer_body.get("answer", ""))
    learn_rule = bool(answer_body.get("learn_rule", True))
    try:
        return await brain.answer_question(qid, answer_text, learn_rule=learn_rule)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/brain/dismiss/{qid}")
async def api_brain_dismiss(qid: int) -> dict[str, str]:
    storage.resolve_brain_question(qid, "dismissed")
    return {"status": "dismissed"}


@app.get("/api/brain/rules")
async def api_brain_rules() -> dict[str, Any]:
    return {"rules": storage.list_brain_rules()}


@app.put("/api/brain/rules/{rule_id}")
async def api_update_brain_rule(rule_id: int, body: dict[str, Any]) -> dict[str, Any]:
    ok = storage.update_brain_rule(rule_id, str(body.get("pattern", "")), str(body.get("action_policy", "")))
    if not ok:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"status": "updated", "id": rule_id}


@app.delete("/api/brain/rules/{rule_id}")
async def api_delete_brain_rule(rule_id: int) -> dict[str, Any]:
    return {"deleted": storage.delete_brain_rule(rule_id)}


@app.get("/api/plugins")
async def api_plugins() -> dict[str, Any]:
    return {"plugins": list(plugins.tools.keys()), "plugins_dir": str(plugins.plugins_dir)}


@app.get("/api/evolve")
async def api_evolve() -> dict[str, Any]:
    return {
        "enabled": settings.evolve_enabled,
        "auto_apply": settings.evolve_auto_apply,
        "max_cpu": settings.evolve_max_cpu,
        "max_ram": settings.evolve_max_ram,
        "proposals": storage.list_evolve_proposals(limit=20),
    }


@app.post("/api/evolve/run")
async def api_evolve_run() -> dict[str, Any]:
    return await evolver.evolve_once(force=True)


@app.get("/api/evolve/proposals")
async def api_evolve_proposals() -> dict[str, Any]:
    return {"proposals": storage.list_evolve_proposals()}


@app.get("/api/processes")
async def api_processes() -> dict[str, Any]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "ps", "-eo", "pid,%cpu,%mem,comm", "--sort=-%cpu", "-n", "30",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        lines = stdout.decode("utf-8", errors="ignore").strip().splitlines()
        procs = []
        for line in lines[1:]:
            parts = line.strip().split(None, 3)
            if len(parts) == 4:
                procs.append({
                    "pid": int(parts[0]),
                    "cpu": float(parts[1]),
                    "mem": float(parts[2]),
                    "name": parts[3]
                })
        return {"processes": procs}
    except Exception as exc:
        return {"processes": [], "error": str(exc)}


@app.post("/api/processes/{pid}/kill")
async def api_kill_process(pid: int) -> dict[str, Any]:
    try:
        proc = await asyncio.create_subprocess_exec("kill", "-9", str(pid))
        await proc.communicate()
        return {"status": "killed", "pid": pid}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/terminal/run")
async def api_terminal_run(body: TerminalCommand) -> dict[str, Any]:
    cmd = body.command.strip()
    if not cmd:
        return {"stdout": "", "stderr": "Empty command", "returncode": 1}
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        return {
            "stdout": stdout.decode("utf-8", errors="ignore"),
            "stderr": stderr.decode("utf-8", errors="ignore"),
            "returncode": proc.returncode
        }
    except TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        return {"stdout": "", "stderr": "Command timed out (30s limit)", "returncode": -1}
    except Exception as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": -1}


@app.get("/api/logs")
async def api_logs(unit: str = "sysagent.service", lines: int = 50) -> dict[str, Any]:
    clean_lines = str(min(200, max(1, int(lines))))
    try:
        proc = await asyncio.create_subprocess_exec(
            "journalctl", f"--user-unit={unit}", "-n", clean_lines, "--no-pager",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout_bytes, _ = await proc.communicate()
        log_text = stdout_bytes.decode("utf-8", errors="ignore")
        if not log_text.strip():
            proc2 = await asyncio.create_subprocess_exec(
                "journalctl", "-u", unit, "-n", clean_lines, "--no-pager",
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            stdout_bytes2, _ = await proc2.communicate()
            log_text = stdout_bytes2.decode("utf-8", errors="ignore")
        return {"unit": unit, "logs": log_text}
    except Exception as exc:
        return {"unit": unit, "logs": f"Error fetching logs: {exc!s}"}


@app.post("/api/control/{action}")
async def api_quick_control(action: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    from .tools import execute_tool

    body = body or {}
    val = str(body.get("value", ""))
    try:
        if action == "screenshot":
            res = await asyncio.to_thread(execute_tool, "take_screenshot", {}, watcher, storage, plugins)
            if isinstance(res, str) and '"error":' in res:
                raise HTTPException(status_code=500, detail=json.loads(res)["error"])
            return {"status": "success", "message": "Screenshot captured"}
        if action == "volume_set":
            pct = max(0, min(100, int(val or 50)))
            res_str = await asyncio.to_thread(
                execute_tool, "desktop_control", {"action": "volume_set", "value": str(pct)}, watcher, storage, plugins
            )
            data = json.loads(res_str) if isinstance(res_str, str) else {}
            return {"status": "success" if data.get("status") == "ok" else "failed", "value": pct}
        if action == "dnd":
            return {"status": "success", "dnd": val}
        if action == "pomodoro":
            mins = max(1, int(val or 25))
            res_str = await asyncio.to_thread(
                execute_tool, "start_focus_timer", {"minutes": mins, "label": f"Pomodoro ({mins}m)"}, watcher, storage, plugins
            )
            watcher._send_desktop_notification("Pomodoro Started", f"{mins}-minute focus session started.")
            data = json.loads(res_str) if isinstance(res_str, str) else {}
            return {"status": "success", "minutes": mins, "timer_id": data.get("id")}
        raise HTTPException(status_code=400, detail="Unknown control action")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/evolve/apply/{proposal_id}")
async def api_evolve_apply(proposal_id: int) -> dict[str, Any]:
    ok = evolver.apply_proposal(proposal_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Proposal not found or failed tests")
    return {"status": "applied", "id": proposal_id}


@app.post("/api/evolve/reject/{proposal_id}")
async def api_evolve_reject(proposal_id: int) -> dict[str, str]:
    storage.update_evolve_status(proposal_id, "rejected")
    return {"status": "rejected"}


@app.post("/api/evolve/rollback")
async def api_evolve_rollback() -> dict[str, Any]:
    return {"rolled_back": evolver.rollback_last_evolve()}


@app.get("/api/self")
async def api_self_health() -> dict[str, Any]:
    return self_monitor.get_self_health()


@app.get("/api/prompts")
async def api_list_prompts() -> dict[str, Any]:
    return {"prompts": PromptRegistry(storage).list_status()}


@app.post("/api/prompts/evolve")
async def api_prompts_evolve() -> dict[str, Any]:
    return await evolver.evolve_prompts_once()


@app.post("/api/prompts/{key}/rollback")
async def api_prompts_rollback(key: str) -> dict[str, Any]:
    res = PromptRegistry(storage).rollback_prompt(key)
    if res.get("status") == "error":
        raise HTTPException(status_code=404, detail=res.get("error", "Unknown prompt key"))
    return res


@app.post("/api/self/maintain")
async def api_self_maintain() -> dict[str, Any]:
    return await asyncio.to_thread(self_monitor.run_maintenance)


@app.get("/api/local")
async def api_local_status() -> dict[str, Any]:
    from .local_mind import LocalMind

    return LocalMind(storage).get_status()


@app.post("/api/local/train")
async def api_local_train(run_lora: bool = False) -> dict[str, Any]:
    from .local_mind import LocalMind

    return await asyncio.to_thread(LocalMind(storage).train, run_lora)


@app.post("/api/local/download")
async def api_local_download(preset: str = "qwen2.5-1.5b") -> dict[str, Any]:
    from .local_mind import download_default_gguf

    path = await asyncio.to_thread(download_default_gguf, preset)
    return {"status": "downloaded", "preset": preset, "path": str(path)}


@app.post("/api/local/distill")
async def api_local_distill(count_per_teacher: int = 2) -> dict[str, Any]:
    from .local_mind import LocalMind

    return await asyncio.to_thread(LocalMind(storage).distill_from_teachers, count_per_teacher)


@app.get("/api/neural")
async def api_neural_status() -> dict[str, Any]:
    from .neural import NeuralEngine

    return NeuralEngine(storage).get_status()


@app.post("/api/neural/train")
async def api_neural_train() -> dict[str, Any]:
    from .neural import NeuralEngine

    return await asyncio.to_thread(NeuralEngine(storage).train)


@app.post("/api/neural/pulse")
async def api_neural_pulse(body: dict[str, Any] | None = None) -> dict[str, Any]:
    from .neural import NeuralEngine

    q = str((body or {}).get("query") or "system status")
    return await asyncio.to_thread(NeuralEngine(storage).pulse, q, watcher.get_health_snapshot())


@app.get("/api/self/export")
async def api_self_export() -> dict[str, Any]:
    return export_brain_bundle(storage)


@app.post("/api/self/import")
async def api_self_import(bundle: dict[str, Any]) -> dict[str, int]:
    counts = import_brain_bundle(bundle, storage)
    plugins.reload()
    return counts


@app.get("/api/plans")
async def api_plans() -> dict[str, Any]:
    return {"plans": storage.list_goal_plans(limit=15)}


@app.get("/api/lessons")
async def api_lessons() -> dict[str, Any]:
    return {"lessons": storage.list_lessons(limit=20)}


@app.post("/api/undo-batch")
async def api_undo_batch(batch_id: str | None = None) -> dict[str, Any]:
    from .tools import execute_tool

    raw = execute_tool(
        "undo_batch_actions",
        {"batch_id": batch_id} if batch_id else {},
        watcher,
        storage,
        plugins=plugins,
    )
    return json.loads(raw) if isinstance(raw, str) else {"result": raw}


@app.get("/api/search")
async def api_search(q: str = "") -> dict[str, Any]:
    term = q.strip()
    results: list[dict[str, Any]] = []
    if term:
        low = term.lower()
        for s in storage.list_sessions(20):
            if low in s["title"].lower():
                results.append({"type": "session", "title": s["title"], "detail": s["session_id"]})
        results.extend(storage.search_knowledge(term, limit=25))
    return {"results": results[:25]}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    if settings.auth_token:
        tok = websocket.query_params.get("token", "")
        if tok != settings.auth_token:
            await websocket.close(code=4401)
            return
    await websocket.accept()
    session_id = websocket.query_params.get("session_id", "default")
    turn_task: asyncio.Task[None] | None = None

    async def emit(payload: dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except (WebSocketDisconnect, RuntimeError):
            pass

    async def handle_message(text: str, locale_override: str | None = None) -> None:
        import traceback

        cancelled = False
        try:
            await agent.run_turn(session_id, text, emit=emit, locale=locale_override)
        except asyncio.CancelledError:
            cancelled = True
            for aid, pend in list(agent.pending_approvals.items()):
                if pend.session_id == session_id:
                    agent.resolve_approval(aid, False)
        except Exception as exc:
            tb_lines = traceback.format_exc().strip().splitlines()
            loc_hint = next((ln.strip() for ln in reversed(tb_lines) if ln.strip().startswith("File ")), "")
            storage.add_brain_thought("ignore", 0, f"Chat runtime error ({loc_hint}): {exc}")
            await emit({"type": "error", "message": f"{exc} ({loc_hint})" if loc_hint else str(exc)})
        finally:
            await emit({"type": "done", "cancelled": cancelled} if cancelled else {"type": "done"})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await emit({"type": "error", "message": "Invalid JSON message."})
                continue
            if not isinstance(data, dict):
                await emit({"type": "error", "message": "Message must be a JSON object."})
                continue
            msg_type = data.get("type", "message")

            if msg_type == "approval_response":
                approval_id = str(data.get("approval_id", ""))
                approved = bool(data.get("approved", False))
                agent.resolve_approval(approval_id, approved)
            elif msg_type == "cancel":
                for aid, pend in list(agent.pending_approvals.items()):
                    if pend.session_id == session_id:
                        agent.resolve_approval(aid, False)
                if turn_task and not turn_task.done():
                    turn_task.cancel()
                else:
                    await emit({"type": "done", "cancelled": True})
            elif msg_type == "set_model":
                new_model = str(data.get("model") or "").strip()
                if not new_model:
                    await emit({"type": "error", "message": "Model name is required."})
                else:
                    agent._session_models[session_id] = new_model
                    if hasattr(agent, "_pinned_models"):
                        agent._pinned_models[session_id] = new_model
                    if data.get("persist_default"):
                        settings.model = new_model
                    await emit({"type": "system", "message": f"Model set to {new_model}"})
            elif msg_type == "reset":
                agent.reset_session(session_id)
                await emit({"type": "system", "message": t("chat_reset", settings.locale)})
            elif msg_type == "message":
                user_text = str(data.get("text", "")).strip()
                if not user_text:
                    continue
                if turn_task and not turn_task.done():
                    await emit(
                        {"type": "error", "message": t("prev_task_running", settings.locale)}
                    )
                    continue
                locale_override = data.get("locale")
                turn_task = asyncio.create_task(handle_message(user_text, locale_override))
    except WebSocketDisconnect:
        if turn_task and not turn_task.done():
            turn_task.cancel()


@app.websocket("/ws/evolve")
async def ws_evolve(websocket: WebSocket) -> None:
    if settings.auth_token:
        tok = websocket.query_params.get("token", "")
        if tok != settings.auth_token:
            await websocket.close(code=4401)
            return
    await websocket.accept()

    async def emit_event(payload: dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except (WebSocketDisconnect, RuntimeError):
            pass

    try:
        res = await evolver.evolve_once(force=True, emit_event=emit_event)
        await emit_event({"type": "result", "data": res})
    except Exception as exc:
        await emit_event({"type": "error", "message": str(exc)})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
