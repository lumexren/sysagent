"""AI, Brain, Self-Evolution, LocalMind, and Neural CLI subcommands."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import websockets

from .config import get_settings
from .i18n import t
from .ipc import ipc_get, ipc_post, ipc_ws_connect
from .storage import Storage


def _auth_headers() -> dict[str, str]:
    tok = get_settings().auth_token
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _cmd_brain(base_url: str, action: str, locale: str | None = None) -> None:
    loc = locale or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "think":
            resp = ipc_post(f"{base_url}/api/brain/think", headers=hdrs, timeout=30)
            resp.raise_for_status()
            print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
            return
        if action == "digest":
            resp = ipc_get(f"{base_url}/api/brain/digest", headers=hdrs, timeout=30)
            resp.raise_for_status()
            print(resp.json().get("digest", ""))
            return
        resp = ipc_get(f"{base_url}/api/brain", headers=hdrs, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if action == "rules":
            for r in data.get("rules", []):
                print(f"#{r['id']} [{r['confidence']}% | {r['hits']}x] {r['pattern']} -> {r['action_policy']}")
            return
        if action == "log":
            from rich import box
            from rich.console import Console
            from rich.table import Table

            tbl = Table(title="Brain Thoughts", title_style="bold cyan", box=box.ROUNDED, border_style="cyan", header_style="bold")
            tbl.add_column("ID", justify="right")
            tbl.add_column("Decision")
            tbl.add_column("Conf", justify="right")
            tbl.add_column("Summary")
            for th in (data.get("thoughts") or [])[:15]:
                tbl.add_row(str(th.get("id", "")), str(th.get("decision", "")), f"{th.get('confidence', 0)}%", str(th.get("summary", ""))[:80])
            Console().print(tbl)
            return
        print(
            f"Brain enabled={data['enabled']} threshold={data['confidence_threshold']}% "
            f"pending_questions={len(data['questions'])} rules={len(data['rules'])}"
        )
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


async def _stream_evolve(base_url: str) -> None:
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    tok = get_settings().auth_token
    ws_url = f"{ws_url}/ws/evolve" + (f"?token={tok}" if tok else "")
    async with ipc_ws_connect(ws_url, open_timeout=10, close_timeout=10) as ws:
        in_thought = False
        while True:
            try:
                raw = await ws.recv()
            except websockets.ConnectionClosed:
                break
            ev = json.loads(raw)
            if ev.get("type") == "heartbeat":
                print(f"\r\033[K\033[90m{ev['elapsed']}s: analyzing\033[0m", end="", flush=True)
            elif ev.get("type") == "thought":
                print("\r\033[K", end="")
                if not in_thought:
                    print("\033[90m", end="", flush=True)
                    in_thought = True
                print(ev["text"], end="", flush=True)
            elif ev.get("type") == "detail":
                print("\r\033[K", end="")
                if in_thought:
                    print("\033[0m")
                    in_thought = False
                print(f"\033[90m{ev['text']}\033[0m", flush=True)
            elif ev.get("type") == "progress":
                print("\r\033[K", end="")
                if in_thought:
                    print("\033[0m")
                    in_thought = False
                print(f"\033[36m{ev['message']}\033[0m", flush=True)
            elif ev.get("type") == "result":
                print("\r\033[K", end="")
                if in_thought:
                    print("\033[0m")
                    in_thought = False
                print(json.dumps(ev["data"], indent=2, ensure_ascii=False))
            elif ev.get("type") == "error":
                print("\r\033[K", end="")
                if in_thought:
                    print("\033[0m")
                    in_thought = False
                print(f"\033[31mError: {ev['message']}\033[0m", file=sys.stderr)


def _cmd_evolve(base_url: str, action: str, locale: str | None = None) -> None:
    loc = locale or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "run":
            asyncio.run(_stream_evolve(base_url))
            return
        resp = ipc_get(f"{base_url}/api/evolve", headers=hdrs, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if action == "log":
            from rich import box
            from rich.console import Console
            from rich.table import Table

            tbl = Table(title="Evolve Proposals", title_style="bold cyan", box=box.ROUNDED, border_style="cyan", header_style="bold")
            tbl.add_column("ID", justify="right")
            tbl.add_column("Status")
            tbl.add_column("File")
            tbl.add_column("Title")
            for p in (data.get("proposals") or [])[:15]:
                tbl.add_row(str(p.get("id", "")), str(p.get("status", "")), Path(str(p.get("file_path", ""))).name, str(p.get("title", ""))[:65])
            Console().print(tbl)
            return
        print(
            f"Self-Evolve enabled={data['enabled']} auto_apply={data['auto_apply']} "
            f"(max_cpu={data['max_cpu']}%, max_ram={data['max_ram']}%) proposals={len(data['proposals'])}"
        )
        for p in data.get("proposals", [])[:10]:
            print(f"#{p['id']} [{p['status']} | {p['category']}] {p['title']}")
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_prompts(base_url: str, action: str, key: str | None = None, locale: str | None = None) -> None:
    loc = locale or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "evolve":
            resp = ipc_post(f"{base_url}/api/prompts/evolve", headers=hdrs, timeout=120)
            resp.raise_for_status()
            print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
            return
        if action == "rollback":
            if not key:
                print(f"{t('error_prefix', loc)}: prompt key is required for rollback (e.g. agent_main)", file=sys.stderr)
                sys.exit(1)
            resp = ipc_post(f"{base_url}/api/prompts/{key}/rollback", headers=hdrs, timeout=10)
            resp.raise_for_status()
            print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
            return
        resp = ipc_get(f"{base_url}/api/prompts", headers=hdrs, timeout=5)
        resp.raise_for_status()
        prompts = resp.json().get("prompts", [])
        for p in prompts:
            if key and p["key"] != key:
                continue
            vars_str = ", ".join(p.get("required_vars") or []) or "-"
            succ = p.get("successes", 0)
            fail = p.get("failures", 0)
            print(
                f"{p['key']} (v{p['version']}) [{p['reason']}] "
                f"ok={succ} fail={fail} vars=({vars_str})"
            )
            if key:
                print("---\n" + p.get("template", ""))
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_local(base_url: str, action: str, query: str | None = None, locale: str | None = None) -> None:
    from .local_mind import GGUF_PRESETS, LocalMind, download_default_gguf, find_gguf_model

    loc = locale or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "train":
            with Storage() as st_db:
                mind = LocalMind(st_db)
                cb = mind.train_on_own_codebase(force=(query or "").strip().lower() == "codebase")
                st = mind.train(run_lora=True)
            st["codebase_sync"] = {"status": cb.get("status"), "samples": cb.get("samples", 0)}
            print(json.dumps(st, indent=2, ensure_ascii=False))
            return
        if action == "distill":
            with Storage() as st_db:
                mind = LocalMind(st_db)
                cnt = int(query) if (query and query.isdigit()) else 2
                print(f"Running Multi-Teacher Knowledge Distillation ({cnt} samples per teacher)...")
                res = mind.distill_from_teachers(count_per_teacher=cnt)
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return
        if action == "download":
            from .local_mind import ROLE_DEFAULT_PRESETS, SPECIALIST_ROLES, find_all_gguf_models

            installed_names = {p.name for p in find_all_gguf_models()}
            target = (query or "qwen2.5-1.5b").strip().lower()
            if target == "list":
                print("6 Specialist Use-Case Models:")
                for r in SPECIALIST_ROLES:
                    pk = ROLE_DEFAULT_PRESETS[r]
                    spec = GGUF_PRESETS[pk]
                    mark = "[installed]" if find_gguf_model(r, strict=True) else "[missing]  "
                    print(
                        f"  {r:<11} {mark} preset={pk:<22} ~{spec['size_mb']} MB  "
                        f"({spec.get('hf_source', spec['filename'])})"
                    )
                print("\nAll Presets:")
                for k, spec in GGUF_PRESETS.items():
                    mark = "*" if spec["filename"] in installed_names else " "
                    print(f" {mark} {k:<22} role={spec['role']:<11} ~{spec['size_mb']} MB  ({spec['filename']})")
                return
            if target == "all-specialists":
                for r in SPECIALIST_ROLES:
                    pk = ROLE_DEFAULT_PRESETS[r]
                    print(f"Downloading {r} specialist ({pk})...")
                    dest = download_default_gguf(pk)
                    print(f"Saved: {dest}")
                return
            if target == "ensemble":
                existing_coder = find_gguf_model("coding")
                targets = [ROLE_DEFAULT_PRESETS["chat"]] if existing_coder else [ROLE_DEFAULT_PRESETS["coding"], ROLE_DEFAULT_PRESETS["chat"]]
                if existing_coder:
                    print(f"Coding specialist already present: {existing_coder}")
                for p_name in targets:
                    print(f"Downloading ensemble specialist '{p_name}'...")
                    dest = download_default_gguf(p_name)
                    print(f"Saved: {dest}")
                return
            preset = ROLE_DEFAULT_PRESETS.get(target, target)
            print(f"Downloading GGUF '{preset}'...")
            dest = download_default_gguf(preset)
            print(f"Downloaded GGUF model to: {dest}")
            return
        if action == "prune":
            from .local_mind import prune_superseded_models

            dry = (query or "").strip().lower() in {"dry", "dry-run", "preview"}
            removed = prune_superseded_models(dry_run=dry)
            print(json.dumps({"dry_run": dry, "removed": removed}, indent=2, ensure_ascii=False))
            return
        if action == "profile":
            from .self_learn import discover_system_profile

            prof = discover_system_profile(Storage(), seed=True, force=True)
            print(json.dumps(prof, indent=2, ensure_ascii=False))
            return
        if action == "curriculum":
            try:
                resp = ipc_get(f"{base_url}/api/local/curriculum", headers=hdrs, timeout=10)
                resp.raise_for_status()
                cur = resp.json()
            except Exception:
                with Storage() as st_db:
                    cur = LocalMind(st_db).build_distillation_curriculum()
            print(json.dumps(cur, indent=2, ensure_ascii=False))
            return
        if action == "test":
            mind = LocalMind(Storage())
            q = query or "check system status cpu ram"
            tool, args, score = mind.predict_tool(q)
            print(f"Query         : {q}")
            print(f"Predicted Tool: {tool} (score={score}) args={args}")
            print(f"Offline Reply : {mind.synthesize_reply(q)}")
            return
        if action == "bench":
            from rich import box
            from rich.console import Console
            from rich.table import Table

            from .bench import benchmark_local_models

            report = benchmark_local_models(role_filter=query, storage=Storage())
            tbl = Table(title="Local GGUF Model Benchmark", box=box.ROUNDED, border_style="cyan")
            tbl.add_column("Role", style="bold cyan")
            tbl.add_column("Model")
            tbl.add_column("Size", justify="right")
            tbl.add_column("Latency", justify="right")
            tbl.add_column("tok/s", justify="right", style="bold green")
            tbl.add_column("RAM Δ", justify="right")
            tbl.add_column("Status")
            for row in report.get("results") or []:
                lat = f"{row['latency_s']}s" if row.get("latency_s") is not None else "-"
                tps = f"{row['tok_per_s']}" if row.get("tok_per_s") is not None else "-"
                tbl.add_row(
                    str(row.get("role", "")),
                    str(row.get("model", "")),
                    f"{row.get('size_gb', 0)} GB",
                    lat,
                    tps,
                    f"+{row.get('ram_delta_mb', 0)} MB",
                    str(row.get("status", "")),
                )
            console = Console()
            console.print(tbl)
            if report.get("fastest_model"):
                console.print(
                    f"[bold green]Fastest:[/] {report['fastest_role']} ({report['fastest_model']})"
                )
            return
        try:
            resp = ipc_get(f"{base_url}/api/local", headers=hdrs, timeout=5)
            resp.raise_for_status()
            st = resp.json()
        except Exception:
            st = LocalMind(Storage()).get_status()
        print(json.dumps(st, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


def _cmd_neural(base_url: str, action: str, query: str | None = None, locale: str | None = None) -> None:
    from .neural import NeuralEngine

    loc = locale or get_settings().locale
    hdrs = _auth_headers()
    try:
        if action == "train":
            try:
                resp = ipc_post(f"{base_url}/api/neural/train", headers=hdrs, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                data = NeuralEngine(Storage()).train()
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return
        if action in {"pulse", "activate"}:
            q = (query or "check system status and ram usage").strip()
            try:
                resp = ipc_post(f"{base_url}/api/neural/pulse", json={"query": q}, headers=hdrs, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                data = NeuralEngine(Storage()).pulse(q)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return
        try:
            resp = ipc_get(f"{base_url}/api/neural", headers=hdrs, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            data = NeuralEngine(Storage()).get_status()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as exc:
        print(f"{t('error_prefix', loc)}: {exc}", file=sys.stderr)
        sys.exit(1)


