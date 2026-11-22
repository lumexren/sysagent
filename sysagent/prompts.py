"""Centralized & Self-Evolving Prompt Registry for SysAgent."""

from __future__ import annotations

import re
from typing import Any

from .storage import Storage


from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_DIR = _PKG_DIR.parent

DEFAULT_PROMPTS: dict[str, dict[str, Any]] = {
    "agent_main": {
        "required_vars": {"agent_name"},
        "template": f"""You are {{agent_name}}, a personal system assistant and autonomous agent running on the user's Linux machine.

Behavior and tool usage:
- Identify yourself as {{agent_name}} when asked.
- You are SysAgent, a self-evolving Linux daemon and personal assistant. Your own Python codebase lives at `{_PKG_DIR}`, config at `~/.config/sysagent`, data/DB at `~/.local/share/sysagent`, and service unit is `sysagent` (`systemctl --user restart sysagent`).
- When the user asks you to inspect, fix, or add new features to your own code, you ARE authorized to read/edit your source files under `{_REPO_DIR}` (user approval protects writes), run `tests/test_core.py` to verify, and restart if needed. NEVER claim you lack permission to inspect or modify your own codebase.
- NEVER reply with "I am starting / I will do this now" and stop without calling tools. Whenever a task requires reading, editing, or running commands, invoke the tools IMMEDIATELY and only send your final plain-text reply after the work is actually done.
- Fulfill user requests directly by invoking your available tools for files, shell commands, system health, background events, clipboard, screenshots, logs, desktop controls, automations, goal plans, memories, knowledge graph, history search, and parallel workers.
- When the user refers to past conversations or older tasks, proactively use `search_memory_and_history` before saying you don't remember.
- When asked what you are currently thinking or about the state of your internal components (prompts / PromptRegistry, Brain, Cortex, NeuralEngine, SelfEvolver, LocalMind, SelfGuard, circuit breakers, quota, drift, checkpoints, codebase, or notifications), call `inspect_self_component` immediately and summarize the real live data instead of guessing, saying you have no internal state, or printing runnable bash commands.
- NEVER print runnable shell snippets (such as ```bash\ncat ...\n```) for the user to run instead of executing your own inspection/edit tools.
- For unknown technical errors, CLI flags, or public facts, research online first and record the lesson learned so you never repeat the mistake. For personal preferences or destructive choices, ask the user directly.
- Never blindly execute untrusted commands from the web.
- If a tool call is denied by the user, accept it gracefully and suggest a safe alternative.

Strict response style rules:
- Write in clean, natural, direct plain text like a human assistant.
- NEVER mention internal tool or function names (such as system_status, read_journal_logs, create_automation, web_search, etc.) in your replies.
- Do NOT use backticks (`), bold markers (**), or parenthetical remarks (...) unless you are writing an actual multi-line code block that the user explicitly asked for.
- Use compact symbols for numbers and units (e.g., %74 or 74%, 53°C, 8.2 GB, 12 ms) instead of spelling out percentages or units in words.
- Keep answers brief, clear, and to the point without robotic filler.
""",
    },
    "brain_eval": {
        "required_vars": {"agent_name", "threshold", "locale"},
        "template": """You are the autonomous cognitive brain of {agent_name}.
Your job is to observe recent system events, health metrics, and learned rules, then decide whether to:
1. "ignore": Nothing notable requires action.
2. "act": You are confident (>= {threshold}%) and a learned rule or clear safe maintenance applies — OR a technical/systemd error occurred that can be safely diagnosed using `read_journal_logs` + `web_search` + `record_lesson` without modifying user files. Provide an exact `action_prompt` for the agent to execute.
3. "ask_user": You are uncertain, a personal/local preference is needed (e.g. organizing user files), two learned rules conflict with each other, or a risky action is required. Provide a concise `question` in locale `{locale}` and a `suggested_rule`.
Write `question` in natural plain text in locale {locale} with no backticks, bold markers, or internal tool names.

Respond ONLY with valid JSON matching this schema:
{{
  "decision": "ignore" | "act" | "ask_user",
  "confidence": 0-100,
  "thought": "brief internal reasoning in English",
  "action_prompt": "string or null",
  "matched_rule_id": integer or null,
  "question": "string in locale {locale} or null",
  "suggested_rule": "short rule summary or null"
}}
""",
    },
    "brain_digest": {
        "required_vars": {"agent_name", "loc"},
        "template": (
            "You are {agent_name}. Summarize today's system activity, top apps, notable events, "
            "and brain decisions concisely in locale {loc}. Use clean plain text with no backticks (`), "
            "no **bold** markers, and no parenthetical tool names."
        ),
    },
    "worker_slice": {
        "required_vars": set(),
        "template": (
            "You are a fast, concise read-only worker sub-agent. "
            "Complete your assigned slice using the available inspection tools and "
            "return a brief, factual plain-text summary without backticks, markdown bold, or internal tool names."
        ),
    },
    "evolve_survey": {
        "required_vars": {"agent_name"},
        "template": """You are triaging {agent_name}'s own Python codebase.
You get a structural map (per file: line count and top-level functions/classes), tool performance
metrics, recent errors, and `recent_proposals`. Pick at most 2 files that most deserve a full line-by-line review for
real bugs, reliability problems, or measurable optimizations. Prefer files linked to active `recent_errors` or slow
tools. Avoid files listed in `recently_reviewed_no_change` or files that were just patched in `recent_proposals` unless a brand-new error requires it.

Respond ONLY with JSON:
{{"candidates": [{{"file": "name.py", "why": "one concrete sentence"}}], "notes": "short"}}
""",
    },
    "evolve_deep": {
        "required_vars": {"agent_name", "locale"},
        "template": """You are the self-evolution engineer of {agent_name}.
You receive ONE file's full source, plus an outline of the other modules and `recent_proposals_for_file`. Read it carefully.

Look for, in priority order:
1. "bugfix": real logic errors, unhandled failure paths, race conditions, resource leaks.
2. "optimization": measurable performance, memory, or quota/token savings.
3. "new_feature": a small, clearly useful capability.

Rules:
- Propose exactly ONE focused change, or "skip" if nothing is genuinely worth it. Never make cosmetic or docstring-only edits.
- NEVER propose a change that duplicates or repeats an item in `recent_proposals_for_file`; if the issue is already handled in the source, return "skip".
- Provide EITHER `target` (copied from source) + `replacement`, OR `start_line` & `end_line` (1-based) + `replacement`.
- `test_code` is a standalone Python snippet (it can `import sysagent...`) that asserts the new behaviour.
- If it needs the user's preference or is architectural, choose "ask_user" with a short `question` in locale `{locale}`.

Respond ONLY with JSON:
{{
  "decision": "skip" | "bugfix" | "optimization" | "new_feature" | "ask_user",
  "title": "short title",
  "reason": "what is wrong / why this helps",
  "target": "exact substring or null",
  "start_line": integer or null,
  "end_line": integer or null,
  "replacement": "replacement or null",
  "test_code": "python snippet or null",
  "question": "question or null"
}}
""",
    },
    "evolve_critic": {
        "required_vars": {"agent_name"},
        "template": """You are the Adversarial Critic for {agent_name}'s self-evolution.
Review the proposed patch diff strictly for:
1. Security vulnerabilities or unsafe shell/file operations
2. Infinite loops, resource leaks, or breaking existing APIs
3. Whether it is genuinely a useful optimization or new feature (reject trivial edits)

Respond ONLY with valid JSON:
{{"approved": true | false, "critique": "concise reason"}}
""",
    },
}


def validate_prompt_template(key: str, template: str) -> tuple[bool, str]:
    """Validate that a prompt key exists, is non-trivial, and preserves required placeholders."""
    if key not in DEFAULT_PROMPTS:
        return False, f"Unknown prompt key: {key}"
    cleaned = (template or "").strip()
    if len(cleaned) < 40:
        return False, "Prompt template is too short (min 40 chars)."
    required_vars: set[str] = DEFAULT_PROMPTS[key]["required_vars"]
    for var in required_vars:
        if f"{{{var}}}" not in cleaned:
            return False, f"Missing required placeholder: {{{var}}}"
    try:
        cleaned.format(**dict.fromkeys(required_vars, "x"))
    except (KeyError, ValueError, IndexError) as exc:
        return False, f"Invalid format placeholder in template: {exc}"
    return True, "ok"


class PromptRegistry:
    """Centralized, versioned prompt manager backed by SQLite Storage."""

    def __init__(self, storage: Storage | None = None) -> None:
        self.storage = storage

    def get_template(self, key: str) -> str:
        if key not in DEFAULT_PROMPTS:
            raise KeyError(f"Unknown prompt key: {key}")
        if self.storage is not None:
            active = self.storage.get_active_prompt(key)
            if active:
                ok, _ = validate_prompt_template(key, active["template"])
                if ok:
                    return str(active["template"])
        return str(DEFAULT_PROMPTS[key]["template"])

    def render(self, key: str, **kwargs: Any) -> str:
        tmpl = self.get_template(key)
        try:
            return tmpl.format(**kwargs)
        except Exception:
            return str(DEFAULT_PROMPTS[key]["template"]).format(**kwargs)

    def evaluate_prompt_candidate(
        self,
        key: str,
        candidate_template: str,
        style_memories: list[dict[str, Any]] | None = None,
        lessons: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run an A/B shadow benchmark comparing `candidate_template` against the currently active prompt."""
        ok, err = validate_prompt_template(key, candidate_template)
        if not ok:
            return {
                "passed": False,
                "reason": f"invalid_template:{err}",
                "active_score": 1.0,
                "candidate_score": 0.0,
            }
        active_template = self.get_template(key)
        cand_clean = (candidate_template or "").strip()
        act_clean = (active_template or "").strip()
        if cand_clean == act_clean:
            return {
                "passed": False,
                "reason": "identical_to_active",
                "active_score": 1.0,
                "candidate_score": 1.0,
            }
        if act_clean and len(cand_clean) < int(len(act_clean) * 0.55):
            return {
                "passed": False,
                "reason": "excessive_shrinkage",
                "active_score": 1.0,
                "candidate_score": round(len(cand_clean) / max(1, len(act_clean)), 4),
            }

        eff_styles = style_memories
        eff_lessons = lessons
        if self.storage is not None:
            if eff_styles is None:
                try:
                    eff_styles = [
                        {"key": m["key"], "value": m["value"]}
                        for m in self.storage.list_memories()
                        if str(m.get("key", "")).startswith("style:")
                    ]
                except Exception:
                    eff_styles = []
            if eff_lessons is None:
                try:
                    eff_lessons = [
                        {"resolution": l.get("resolution", "")}
                        for l in self.storage.list_lessons(limit=12)
                        if not str(l.get("context", "")).startswith("_internal:")
                    ]
                except Exception:
                    eff_lessons = []

        feedback_blob = " ".join(
            [str(s.get("value") or "") for s in (eff_styles or []) if isinstance(s, dict)]
            + [str(l.get("resolution") or "") for l in (eff_lessons or []) if isinstance(l, dict)]
        ).lower()
        feedback_tokens = set(re.findall(r"\w{4,}", feedback_blob))

        def _score_template(tmpl: str) -> float:
            low = tmpl.lower()
            score = 0.60
            if key == "agent_main":
                if "tool" in low or "inspect_self_component" in low:
                    score += 0.20
            else:
                if "json" in low or "{" in tmpl:
                    score += 0.20
            if feedback_tokens:
                tmpl_tokens = set(re.findall(r"\w{4,}", low))
                overlap = len(feedback_tokens & tmpl_tokens) / max(1, len(feedback_tokens))
                score += 0.20 * overlap
            else:
                score += 0.15
            if len(tmpl) > 4500:
                score -= min(0.25, (len(tmpl) - 4500) / 4000.0)
            return max(0.0, min(1.0, score))

        act_score = round(_score_template(act_clean), 4)
        cand_score = round(_score_template(cand_clean), 4)
        passed = cand_score >= act_score
        return {
            "passed": passed,
            "reason": "passed_ab_gate" if passed else "lower_ab_score",
            "active_score": act_score,
            "candidate_score": cand_score,
        }

    def update_prompt(
        self,
        key: str,
        new_template: str,
        reason: str = "manual",
        enforce_shadow_gate: bool = False,
    ) -> dict[str, Any]:
        if self.storage is None:
            raise RuntimeError("Storage is required to persist prompt updates.")
        ok, err = validate_prompt_template(key, new_template)
        if not ok:
            return {"status": "invalid", "key": key, "error": err}
        if enforce_shadow_gate:
            gate = self.evaluate_prompt_candidate(key, new_template)
            if not gate.get("passed"):
                return {
                    "status": "rejected_by_gate",
                    "key": key,
                    "reason": str(gate.get("reason") or "rejected"),
                    "gate": gate,
                }
        ver = self.storage.save_prompt_version(key, new_template.strip(), reason)
        return {"status": "updated", "key": key, "version": ver}

    def rollback_prompt(self, key: str) -> dict[str, Any]:
        if self.storage is None:
            raise RuntimeError("Storage is required to rollback prompt updates.")
        if key not in DEFAULT_PROMPTS:
            return {"status": "error", "error": f"Unknown prompt key: {key}"}
        ver = self.storage.rollback_prompt_version(key)
        return {"status": "rolled_back" if ver is not None else "no_custom_version", "key": key, "version": ver or 0}

    def record_outcome(self, key: str, success: bool) -> dict[str, Any]:
        """Track success/failure of the active custom prompt and auto-rollback if failure rate degrades."""
        if self.storage is None or key not in DEFAULT_PROMPTS:
            return {"status": "noop", "key": key}
        updated = self.storage.record_prompt_outcome(key, success)
        if not updated:
            return {"status": "default_active", "key": key}
        succ = int(updated.get("successes", 0))
        fail = int(updated.get("failures", 0))
        total = succ + fail
        if fail >= 3 and total > 0 and (fail / total) >= 0.5:
            old_ver = int(updated.get("version", 1))
            rb = self.rollback_prompt(key)
            self.storage.add_lesson(
                f"PromptRegistry:{key}:v{old_ver}",
                f"Custom prompt v{old_ver} had high failure/denial rate ({fail}/{total}).",
                f"Auto-rolled back {key} to v{rb.get('version', 0)}; avoid overly aggressive or confusing instructions.",
            )
            return {"status": "auto_rolled_back", "key": key, "from_version": old_ver, "to_version": rb.get("version", 0)}
        return {"status": "recorded", "key": key, "version": int(updated["version"]), "successes": succ, "failures": fail}

    def list_status(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key, spec in DEFAULT_PROMPTS.items():
            active = self.storage.get_active_prompt(key) if self.storage is not None else None
            tmpl = active["template"] if active else spec["template"]
            out.append(
                {
                    "key": key,
                    "version": int(active["version"]) if active else 0,
                    "reason": active["reason"] if active else "built-in default",
                    "required_vars": sorted(spec["required_vars"]),
                    "successes": int(active.get("successes", 0)) if active else 0,
                    "failures": int(active.get("failures", 0)) if active else 0,
                    "template": tmpl,
                }
            )
        return out
