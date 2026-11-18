import shutil
import tempfile
from pathlib import Path

from sysagent.safety import is_auto_trust_allowed
from sysagent.self_learn import (
    check_patch_outcomes,
    error_clusters,
    files_to_deprioritize,
    proposal_stats,
    suggest_workflow_skills,
)
from sysagent.storage import Storage

with tempfile.TemporaryDirectory() as td:
    st = Storage(Path(td) / "t.db")

    # --- proposal stats / bandit
    for _ in range(3):
        st.add_evolve_proposal("/p/sysagent/watcher.py", "bugfix", "t", "r", "", "", False, "failed")
    st.add_evolve_proposal("/p/sysagent/tools.py", "optimization", "t", "r", "", "", True, "applied")
    stats = proposal_stats(st)
    assert stats["success_rate"] == 0.25, stats
    assert files_to_deprioritize(stats) == {"watcher.py"}

    # --- error clusters
    msg = 'Chat runtime error (File "/h/Projects/sysagent/sysagent/agent.py", line 77, in x): boom'
    st.add_brain_thought("ignore", 0, msg)
    assert error_clusters(st) == []
    st.add_brain_thought("ignore", 0, msg)
    cl = error_clusters(st)
    assert cl and cl[0]["file"] == "agent.py" and cl[0]["count"] == 2, cl

    # --- patch outcome regression -> rollback
    st.record_file_backup("evolve", "/p/sysagent/tools.py", True, "old", batch_id="b")
    for _ in range(3):
        st.add_brain_thought("ignore", 0, "Runtime tool error in x (File \"/h/sysagent/sysagent/tools.py\", line 9): bad")
    rolled: list[int] = []
    ev = check_patch_outcomes(st, lambda pid: rolled.append(pid) or True)
    assert ev and ev[0]["outcome"] == "rolled_back" and rolled, ev
    assert check_patch_outcomes(st, lambda pid: True) == []  # already decided

    # --- workflow -> skill suggestion (once)
    for _ in range(3):
        st.record_workflow("check logs and restart", [{"name": "read_journal_logs"}, {"name": "run_command"}])
    assert suggest_workflow_skills(st) == 1
    assert suggest_workflow_skills(st) == 0
    assert any("read_journal_logs -> run_command" in q["question"] for q in st.list_brain_questions("pending"))

# --- auto-trust guard
assert is_auto_trust_allowed("run_command", {"command": "pytest tests"})
assert not is_auto_trust_allowed("run_command", {"command": "python evil.py"})
assert not is_auto_trust_allowed("run_command", {"command": "/usr/bin/rm -r x"})

# --- 6 specialist roles & presets
from sysagent.local_mind import (
    GGUF_PRESETS,
    ROLE_DEFAULT_PRESETS,
    SPECIALIST_ROLES,
    LocalMind,
    classify_gguf_role,
    classify_task_role,
)

assert len(SPECIALIST_ROLES) == 6
for role in SPECIALIST_ROLES:
    pk = ROLE_DEFAULT_PRESETS[role]
    assert pk in GGUF_PRESETS
    assert GGUF_PRESETS[pk]["role"] == role
    assert GGUF_PRESETS[pk].get("hf_source")
    assert classify_gguf_role(Path(GGUF_PRESETS[pk]["filename"])) == role

assert classify_task_role("fix bug in python code") == "coding"
assert classify_task_role("analyze screenshot image", [{"name": "take_screenshot"}]) == "multimodal"
assert classify_task_role("why is this happening analyze") == "reasoning"
assert classify_task_role("hello how are you") == "chat"

with tempfile.TemporaryDirectory() as td:
    st_lm = Storage(Path(td) / "lm.db")
    mind = LocalMind(st_lm)
    status = mind.get_status()
    assert set(status["specialist_roles"].keys()) == set(SPECIALIST_ROLES)
    assert status.get("weakest_role") in SPECIALIST_ROLES
    assert mind.semantic_similarity("check system status cpu", "system status cpu usage") > 0.3

from sysagent.config import get_settings
from sysagent.local_mind import prune_superseded_models
from sysagent.self_learn import (
    auto_rollback_failing_prompts,
    build_self_roadmap,
    compact_distilled_dataset,
    compute_adaptive_schedule,
    deduplicate_semantic_knowledge,
    discover_system_profile,
    inject_plugin_skills_to_local_mind,
    list_immunity_patterns,
    record_immunity_pattern,
    record_shadow_evaluation,
    scan_coverage_gaps,
    shadow_stats,
)
from sysagent.watcher import SystemWatcher

assert isinstance(prune_superseded_models(dry_run=True), list)
with tempfile.TemporaryDirectory() as td_w:
    st_hyg = Storage(Path(td_w) / "w.db")
    gpu_info = SystemWatcher(get_settings(), st_hyg).get_health_snapshot().get("gpu")
    if shutil.which("nvidia-smi"):
        assert gpu_info is not None and gpu_info.get("vram_total_mb", 0) > 0
    # Memory & lesson hygiene
    st_hyg.set_memory("old_path", "/home/eren/.gemini/antigravity/scratch/sysagent/sysagent/x.py")
    st_hyg.add_lesson("ctx", "disk space warning threshold reached on root partition", "res")
    st_hyg.add_lesson("ctx", "disk space warning threshold reached on root partition today", "res")
    hyg = deduplicate_semantic_knowledge(st_hyg)
    assert hyg["stale_memories_removed"] == 1 and hyg["duplicate_lessons_merged"] == 1, hyg
    # Prompt rollback doctor
    v1 = st_hyg.save_prompt_version("agent_main", "bad template {agent_name}", "test")
    assert v1 == 1
    for _ in range(3):
        st_hyg.record_prompt_outcome("agent_main", False)
    assert auto_rollback_failing_prompts(st_hyg) == ["agent_main"]
    # System profiler, immunity, plugin skill injection, adaptive schedule
    prof = discover_system_profile(st_hyg, seed=False, force=True)
    assert prof.get("summary") and prof.get("cpu_cores", 0) >= 1
    assert "hardware" in prof and prof["hardware"].get("cpu_model")
    assert "software" in prof and prof["software"].get("path_binaries_count", 0) > 50
    assert len(prof.get("mounts") or []) >= 1 and "environment" in prof
    record_immunity_pattern(st_hyg, "run wrong command", "run_command", "User denied")
    assert len(list_immunity_patterns(st_hyg)) == 1
    class _FakePlugs:
        def declarations(self):
            return [{"name": "custom_ping_test", "description": "Check ping latency"}]
    import sysagent.local_mind as lm_mod
    orig_df, orig_wf = lm_mod.DISTILLED_FILE, lm_mod.WEIGHTS_FILE
    lm_mod.DISTILLED_FILE = Path(td_w) / "d.jsonl"
    lm_mod.WEIGHTS_FILE = Path(td_w) / "w.json"
    try:
        assert inject_plugin_skills_to_local_mind(st_hyg, _FakePlugs()) == 1
        assert inject_plugin_skills_to_local_mind(st_hyg, _FakePlugs()) == 0
        # Dataset compactor on temp JSONL
        import json as _json
        rows = [_json.dumps({"user": "check system status cpu ram memory", "tool_calls": [{"name": "system_status"}], "reply": "ok"}) for _ in range(45)]
        lm_mod.DISTILLED_FILE.write_text("\n".join(rows) + "\n", encoding="utf-8")
        comp = compact_distilled_dataset(max_per_class=80, sim_threshold=0.92)
        assert comp["before"] >= 45 and comp["removed"] >= 40, comp
    finally:
        lm_mod.DISTILLED_FILE, lm_mod.WEIGHTS_FILE = orig_df, orig_wf
    sched = compute_adaptive_schedule(SystemWatcher(get_settings(), st_hyg), st_hyg)
    assert sched["mode"] in {"active_work", "media_quiet", "deep_sleep_learn", "proactive_guard"}
    # Shadow eval + graduation + coverage gaps + roadmap
    for _ in range(5):
        record_shadow_evaluation(st_hyg, "check system status cpu ram memory", [{"name": "system_status"}])
    sh = shadow_stats(st_hyg)
    assert sh["hits"] + sh["misses"] == 5 and sh["accuracy"] is not None
    cov = scan_coverage_gaps()
    assert cov["total_count"] > 20 and cov["coverage_pct"] > 0, cov
    assert len(build_self_roadmap(st_hyg)) >= 1

# --- 1. Precision AST slicer + async blocking auditor
from sysagent.evolve_ast import extract_ast_focus_slice, module_outline

async_bad_src = """import asyncio, subprocess

async def good_fn():
    def _sync():
        return subprocess.run(["echo", "hi"])
    return await asyncio.to_thread(_sync)

async def bad_fn():
    return subprocess.run(["echo", "bad"])
"""
outline = module_outline(async_bad_src)
blocking = [f for f in outline["ast_findings"] if f.startswith("BUG:async_blocking_call:")]
assert len(blocking) == 1 and "bad_fn" not in blocking[0] and ":L9:subprocess.run" in blocking[0], outline

big_src = "import os\n\ndef alpha():\n" + ("    x = 1\n" * 400) + "\ndef target_bug_fn():\n    return os.getpid()\n" + ("# pad\n" * 600)
sl = extract_ast_focus_slice(big_src, why="Static signature bug in target_bug_fn", max_chars=2500)
assert sl["sliced"] and sl["focus_symbol"] == "target_bug_fn" and len(sl["source"]) <= 2500, sl

# --- 2 & 3. Learned argument templates, tool chain predictor & shadow replay
import sysagent.local_mind as lm_mod
from sysagent.self_learn import (
    auto_forge_workflow_skills,
    heal_runtime_anomalies,
    replay_shadow_history,
    surgical_git_rollback,
    synthesize_coverage_tests,
)

with tempfile.TemporaryDirectory() as td_adv:
    tdp = Path(td_adv)
    st_adv = Storage(tdp / "adv.db")
    orig_paths = (lm_mod.DISTILLED_FILE, lm_mod.WEIGHTS_FILE, lm_mod.ROLE_ADAPTERS_FILE, lm_mod.LORA_TRAIN_FILE)
    lm_mod.DISTILLED_FILE = tdp / "distilled.jsonl"
    lm_mod.WEIGHTS_FILE = tdp / "weights.json"
    lm_mod.ROLE_ADAPTERS_FILE = tdp / "adapters.json"
    lm_mod.LORA_TRAIN_FILE = tdp / "lora.txt"
    try:
        m_adv = LocalMind(st_adv)
        m_adv.record_and_learn(
            "fetch documentation webpage example site",
            [{"name": "fetch_url", "arguments": {"url": "https://example.org", "max_chars": 2000}}],
            "Fetched webpage.",
        )
        for _ in range(3):
            st_adv.record_workflow(
                "system health and events",
                [{"name": "system_status", "arguments": {}}, {"name": "list_events", "arguments": {"limit": 5}}],
            )
            m_adv.record_and_learn(
                "check system status cpu ram memory",
                [{"name": "system_status", "arguments": {}}],
                "System OK.",
            )
        m_adv.train()
        assert "fetch_url" in m_adv.weights.get("learned_arg_templates", {}), m_adv.weights
        ext_args = m_adv.extract_arguments("fetch_url", "fetch example site docs")
        assert ext_args.get("url") == "https://example.org", ext_args
        chain = m_adv.predict_tool_chain("check system status cpu ram memory")
        assert [c["name"] for c in chain[:2]] == ["system_status", "list_events"], chain

        sh_rep = replay_shadow_history(st_adv)
        assert sh_rep["hits"] + sh_rep["misses"] >= 4 and sh_rep["accuracy"] is not None, sh_rep

        # --- 4 & 6b. Auto-forge workflow plugin + quarantine broken plugin
        from sysagent.plugins import PluginRegistry

        pdir = tdp / "plugins"
        pdir.mkdir()
        plugs = PluginRegistry(plugins_dir=pdir, mcp_config_file=tdp / "mcp.json")
        forged = auto_forge_workflow_skills(st_adv, plugins=plugs, min_hits=2, max_new=1)
        assert forged == ["macro_system_status_list_events"], forged
        assert "macro_system_status_list_events" in plugs.tools
        macro_out = plugs.tools["macro_system_status_list_events"].handler({})
        assert "system_status" in macro_out and "list_events" in macro_out, macro_out

        (pdir / "broken_plugin.py").write_text("def broken(:\n", encoding="utf-8")
        st_adv.record_tool_metric("run_command", 2200.0, False)
        healed = heal_runtime_anomalies(st_adv, plugins=plugs)
        assert "broken_plugin.py" in healed["quarantined"] and (pdir / "broken_plugin.py.quarantine").is_file(), healed
        assert "run_command" in healed["tuned_tools"], healed
    finally:
        lm_mod.DISTILLED_FILE, lm_mod.WEIGHTS_FILE, lm_mod.ROLE_ADAPTERS_FILE, lm_mod.LORA_TRAIN_FILE = orig_paths

# --- 5. Autonomous coverage test synthesizer (isolated temp repo)
with tempfile.TemporaryDirectory() as td_cov:
    root = Path(td_cov)
    (root / "sysagent").mkdir()
    (root / "sysagent" / "__init__.py").write_text("", encoding="utf-8")
    (root / "sysagent" / "demo_cov.py").write_text(
        "def helper_alpha(x: int) -> int:\n    return x + 1\n\nclass Box:\n    def helper_beta(self) -> int:\n        return 2\n",
        encoding="utf-8",
    )
    (root / "tests").mkdir()
    res_cov = synthesize_coverage_tests(pkg_dir=root / "sysagent", max_symbols=5)
    assert res_cov["added"] == 2 and (root / "tests" / "test_auto_coverage.py").is_file(), res_cov
    assert res_cov["coverage"]["coverage_pct"] == 100, res_cov

# --- 6a. Surgical multi-commit git rollback (isolated temp git repo)
if shutil.which("git"):
    import subprocess as _sp

    with tempfile.TemporaryDirectory() as td_git:
        gp = Path(td_git)
        pkg = gp / "sysagent"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "a.py").write_text("VAL_A = 1\n", encoding="utf-8")
        (pkg / "b.py").write_text("VAL_B = 1\n", encoding="utf-8")
        _sp.run(["git", "init"], cwd=gp, check=True, capture_output=True)
        _sp.run(["git", "config", "user.email", "t@local"], cwd=gp, check=True, capture_output=True)
        _sp.run(["git", "config", "user.name", "Test"], cwd=gp, check=True, capture_output=True)
        _sp.run(["git", "add", "-A"], cwd=gp, check=True, capture_output=True)
        _sp.run(["git", "commit", "-m", "baseline"], cwd=gp, check=True, capture_output=True)
        (pkg / "a.py").write_text("VAL_A = 999  # buggy\n", encoding="utf-8")
        _sp.run(["git", "add", "-A"], cwd=gp, check=True, capture_output=True)
        _sp.run(["git", "commit", "-m", "evolve(#7): broken a"], cwd=gp, check=True, capture_output=True)
        (pkg / "b.py").write_text("VAL_B = 2  # keep\n", encoding="utf-8")
        _sp.run(["git", "add", "-A"], cwd=gp, check=True, capture_output=True)
        _sp.run(["git", "commit", "-m", "evolve(#8): good b"], cwd=gp, check=True, capture_output=True)
        assert surgical_git_rollback(gp, 7) is True
        assert "VAL_A = 1" in (pkg / "a.py").read_text(encoding="utf-8")
        assert "VAL_B = 2" in (pkg / "b.py").read_text(encoding="utf-8")

# --- 7. 5-Pillar Vectorized Neural Engine
from sysagent.neural import NeuralEngine

with tempfile.TemporaryDirectory() as td_nn:
    nnp = Path(td_nn)
    st_nn = Storage(nnp / "nn.db")
    st_nn.add_evolve_proposal("/p/a.py", "bugfix", "fix mem leak", "r", "", "", True, "applied")
    st_nn.add_evolve_proposal("/p/b.py", "new_feature", "risky rewrite", "r", "", "", False, "failed")
    st_nn.set_memory("nvidia_gpu", "RTX 3060 12GB CUDA driver 570")
    st_nn.add_lesson("nvidia_gpu crash", "Check journalctl nvidia error logs", "restart display manager")
    st_nn.add_kg_edge("nvidia_gpu", "requires_check", "read_journal_logs")

    eng = NeuralEngine(st_nn, npz_file=nnp / "neural_engine.npz", meta_file=nnp / "neural_meta.json")
    nn_samples = [
        {"user": "check system status cpu ram memory", "tool_calls": [{"name": "system_status", "arguments": {}}], "role": "general"},
        {"user": "list files in downloads folder", "tool_calls": [{"name": "list_directory", "arguments": {"path": "~/Downloads"}}], "role": "general"},
        {"user": "fix python bug in agent code", "tool_calls": [{"name": "edit_file", "arguments": {"path": "agent.py"}}], "role": "coding"},
        {"user": "read systemd journal logs error", "tool_calls": [{"name": "read_journal_logs", "arguments": {"lines": 40}}], "role": "reasoning"},
    ] * 4
    st_res = eng.train(samples=nn_samples, epochs=22, lr=0.02)
    assert st_res["param_count"] > 10000, st_res
    assert st_res["last_loss"] < st_res["initial_loss"], st_res

    t_probs_before = eng.predict_tool_probs("check system status cpu ram memory")
    assert max(t_probs_before, key=t_probs_before.get) == "system_status", t_probs_before
    p_sys_before = t_probs_before["system_status"]
    eng.online_step("check system status cpu ram memory", "system_status", reward=-1.0, lr=0.08)
    t_probs_after = eng.predict_tool_probs("check system status cpu ram memory")
    assert t_probs_after["system_status"] < p_sys_before, (p_sys_before, t_probs_after)

    eng.sync_associative_memory()
    hits = eng.spreading_activation("nvidia_gpu cuda driver", top_k=4, hops=2)
    assert hits and eng.get_status()["synapses"] >= 1, (hits, eng.get_status())
    eng.meta.setdefault("synapses", {}).setdefault("test_node_a", {})["test_node_b"] = 0.02
    assert eng.prune_weak_synapses(min_weight=0.08) >= 1

    class _MockWatcher:
        def get_health_snapshot(self):
            return {"cpu_percent": 32.0, "ram_percent": 58.0, "gpu": {"util_percent": 10.0, "vram_used_mb": 2000, "vram_total_mb": 12288}}

    for _ in range(3):
        wf = eng.observe_world_state(_MockWatcher(), active_role="coding")
    assert 0.0 <= wf["bottleneck_prob"] <= 1.0 and "pred_ram_10m" in wf, wf

    pv0 = eng.score_patch_candidate("z.py", "bugfix", "clean minor fix", "safe")
    pv_up = eng.online_patch_step("z.py", "bugfix", "clean minor fix", "safe", success=True, lr=0.1)
    assert pv_up > pv0, (pv0, pv_up)
    pv_down = eng.online_patch_step("z.py", "bugfix", "clean minor fix", "safe", success=False, lr=0.15)
    assert pv_down < pv_up, (pv_up, pv_down)
    pulse_out = eng.pulse("fix python bug in agent code")
    assert pulse_out["hidden_dim"] == 128 and len(pulse_out["role_probs"]) == 6, pulse_out

print("SELF_LEARN_TESTS_OK")
