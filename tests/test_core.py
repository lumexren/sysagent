"""Core regression test suite for SysAgent (executed by CI and SelfEvolver staging)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from sysagent.auth import _AgyCliInteractions
from sysagent.cli import _expand_at_paths, _format_terminal_text, _strip_paste_sequences
from sysagent.config import get_settings
from sysagent.local_mind import LocalMind
from sysagent.plugins import PluginRegistry, validate_plugin_code_isolated
from sysagent.prompts import PromptRegistry, validate_prompt_template
from sysagent.safety import assess_tool_call
from sysagent.self_ops import ToolCache, scrub_secrets, select_relevant_tools
from sysagent.storage import Storage
from sysagent.tools import TOOL_DECLARATIONS, execute_tool
from sysagent.watcher import SystemWatcher
import sysagent.server as srv


def run_all_tests() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        db = Storage(tmp / "core_test.db")

        # 1. Storage WAL, FTS5, Workflows, Consolidation
        assert db._connect().execute("PRAGMA journal_mode;").fetchone()[0].lower() == "wal"
        db.set_memory("style:verbosity", "short")
        db.set_memory("proj_dir", "/tmp/demo")
        db.set_memory("Proj_Dir ", "dup")
        db.add_lesson("ctx", "ins", "res")
        db.add_kg_edge("proj", "uses", "python")
        db.add_message("s1", "user", "proj_dir demo python")
        assert len(db.search_knowledge("demo python", limit=5)) >= 1
        db.record_workflow("check system and memories", [{"name": "system_status"}, {"name": "recall_memories"}])
        assert len(db.find_matching_workflows("check system", limit=2)) == 1
        cons = db.consolidate_memories()
        assert cons["memories_removed"] >= 1

        # 2. PromptRegistry validation, outcome scoring & auto-rollback
        ok, _ = validate_prompt_template("agent_main", "too short")
        assert not ok
        reg = PromptRegistry(db)
        upd = reg.update_prompt("agent_main", "You are {agent_name}, a helpful Linux assistant.", "test")
        assert upd["status"] == "updated"
        reg.record_outcome("agent_main", False)
        reg.record_outcome("agent_main", False)
        rb = reg.record_outcome("agent_main", False)
        assert rb["status"] == "auto_rolled_back"

        # 3. Transactional batch undo isolated from evolve backups
        settings = get_settings()
        watcher = SystemWatcher(settings, storage=db)
        f1 = tmp / "f1.txt"
        f1.write_text("v1", encoding="utf-8")
        execute_tool("write_file", {"path": str(f1), "content": "v2"}, watcher, db, batch_id="b1")
        db.record_file_backup("evolve", str(tmp / "code.py"), True, "orig")
        res_undo = json.loads(execute_tool("undo_batch_actions", {}, watcher, db))
        assert res_undo["status"] == "batch_undone" and f1.read_text(encoding="utf-8") == "v1"

        # 4. Safety, ToolCache, scrub_secrets, select_relevant_tools
        assert assess_tool_call("system_status", {}, mode="smart").requires_approval is False
        assert assess_tool_call("run_command", {"command": "rm -rf /tmp/x"}, mode="smart").requires_approval is True
        scrubbed = json.loads(scrub_secrets(json.dumps({"token": "abcdef1234567890"})))
        assert scrubbed["token"] == "[REDACTED]"
        tc = ToolCache(ttl_sec=60.0, max_entries=2)
        for i in range(4):
            tc.set("read_file", {"path": f"/tmp/{i}"}, str(i))
        assert len(tc._store) == 2
        pruned = select_relevant_tools("hello", TOOL_DECLARATIONS, {d["name"] for d in TOOL_DECLARATIONS})
        pruned_names = {d["name"] for d in pruned}
        assert len(pruned) < len(TOOL_DECLARATIONS)
        assert {"read_journal_logs", "web_search", "fetch_url", "desktop_control", "create_goal_plan", "forge_skill"} <= pruned_names

        # 4b. Multi-brace JSON tool extraction & CLI terminal formatting / @path / paste cleanup
        raw_prose_tc = 'Checking context {meta: 1} before editing:\n{"__tool_calls__": [{"name": "read_file", "arguments": {"path": "app.py"}}]}\nDone.'
        calls, preamble = _AgyCliInteractions._extract_calls(raw_prose_tc)
        assert calls and calls[0]["name"] == "read_file" and "Checking context" in preamble
        fmt = _format_terminal_text("Here is **bold** (`system_status`) and:\n```python\nx = 1\n```")
        assert "\033[1mbold\033[0m" in fmt and "\033[36m```python\nx = 1\n```\033[0m" in fmt and "system_status" not in fmt
        assert _strip_paste_sequences("\x1b[200~line1\nline2\x1b[201~") == "line1\nline2"
        demo_att = tmp / "sample_note.txt"
        demo_att.write_text("hello from file\nsecond line\n", encoding="utf-8")
        exp_txt, exp_lbls = _expand_at_paths("inspect @sample_note.txt please", tmp)
        assert exp_lbls == ["sample_note.txt"] and "hello from file" in exp_txt

        # 5. Isolated plugin validation & MCP config parsing
        bad_ok, _ = validate_plugin_code_isolated("syntax error (", "bad")
        assert not bad_ok
        mcp_cfg = tmp / "mcp.json"
        mcp_cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "echo_srv": {
                            "command": sys.executable,
                            "args": [
                                "-c",
                                (
                                    "import sys, json\n"
                                    "for line in sys.stdin:\n"
                                    "    req = json.loads(line)\n"
                                    "    if req.get('method') == 'tools/list':\n"
                                    "        print(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'tools':[{'name':'ping','description':'pong','inputSchema':{'type':'object','properties':{}}}]}}), flush=True)\n"
                                    "    elif req.get('method') == 'tools/call':\n"
                                    "        print(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{'pong':True}}), flush=True)\n"
                                    "    elif 'id' in req:\n"
                                    "        print(json.dumps({'jsonrpc':'2.0','id':req['id'],'result':{}}), flush=True)\n"
                                ),
                            ],
                            "safe": True,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        plugs = PluginRegistry(tmp / "plugins", mcp_config_file=mcp_cfg)
        assert "mcp_echo_srv_ping" in plugs.tools
        assert plugs.execute("mcp_echo_srv_ping", {}) == {"pong": True}

        # 6. LocalMind training & prediction
        mind = LocalMind(db)
        tool, _, score = mind.predict_tool("check system status cpu ram")
        assert tool == "system_status" and score > 0.2

        # 7. FastAPI endpoints & O(1) memory lookup + system inspection tools
        db.set_memory("k_test", "v_test")
        assert db.get_memory("k_test") == "v_test"
        assert db.get_memory("missing_key") is None

        procs_res = json.loads(execute_tool("list_processes", {"limit": 5}, watcher, db))
        assert isinstance(procs_res["processes"], list) and len(procs_res["processes"]) >= 1
        units_res = json.loads(execute_tool("list_systemd_units", {"limit": 5}, watcher, db))
        assert "units" in units_res
        self_comp = json.loads(execute_tool("inspect_self_component", {"component": "all", "query": "ram"}, watcher, db))
        assert "brain" in self_comp and "neural" in self_comp and "evolve" in self_comp

        from sysagent.neural import NeuralEngine

        nn = NeuralEngine(db, npz_file=tmp / "nn.npz", meta_file=tmp / "nn.json")
        nn.train(epochs=4)
        wf = nn.observe_world_state({"cpu_percent": 25.0, "ram_percent": 55.0})
        assert 0.0 <= wf["pred_ram_10m"] <= 100.0 and "anomaly_score" in wf
        unc = nn.predict_with_uncertainty("check system status cpu ram")
        assert "entropy" in unc and "margin" in unc
        nn.online_transition_step("search and read file", "search_files", "read_file")
        assert len(nn.predict_next_tools("search and read file", prev_tool="search_files", top_k=2)) >= 1
        nn.record_safe_plan("check ram processes", "list_processes", {"sort_by": "memory", "limit": 10})
        matched_plan = nn.match_safe_plan("check ram processes", min_sim=0.85)
        assert matched_plan is not None and matched_plan["tool"] == "list_processes"
        assert nn.infer_arguments("show ram usage processes", "list_processes").get("sort_by") == "memory"
        for _ in range(3):
            nn.online_stall_step("Adımları sırayla uyguluyorum, devam ediyorum.", is_stall=True)
        assert 0.0 <= nn.score_reply_stall("Adımları sırayla uyguluyorum.") <= 1.0

        from sysagent.self_learn import build_causal_timeline, evaluate_graduation_arena, predict_user_routine

        watcher.add_event("health", "warning", "High RAM spike")
        assert isinstance(build_causal_timeline(db, watcher), list)
        assert "graduated" in evaluate_graduation_arena(db)
        _ = predict_user_routine(db)

        rid = db.add_brain_rule("high disk usage in cache", "clean temporary cache", confidence=95)
        for _ in range(5):
            db.reinforce_brain_rule(rid, delta=0)
        assert db.match_fast_action_rule("alert: high disk usage in cache") is not None
        q1 = db.create_brain_question("Should I clean temp cache files now?", "ctx1", "clean_cache")
        q2 = db.create_brain_question("Should I clean temp cache files right now?", "ctx2", "clean_cache")
        assert q1 == q2
        assert db.expire_stale_brain_questions(max_age_hours=0) >= 1
        merged_cues = db.merge_learned_cues({"stall_reply": ["continuing now"], "correction": ["wrong tool"]})
        assert "continuing now" in merged_cues["stall_reply"]
        assert "wrong tool" in db.get_learned_cues()["correction"]

        orig_storage = srv.storage
        srv.storage = db
        try:
            client = TestClient(srv.app)
            assert client.get("/api/status").status_code == 200
            assert len(client.get("/api/prompts").json()["prompts"]) == 7
            assert client.get("/api/local").status_code == 200
        finally:
            srv.storage = orig_storage


if __name__ == "__main__":
    run_all_tests()
    print("CORE_TESTS_OK")
