"""Cognitive brain, self-evolution, knowledge graph, metrics, and prompt storage mixin."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class _BrainStorageMixin:
    """Storage methods for brain rules/thoughts/questions, evolve proposals, KG, metrics, and prompts."""


    def add_brain_rule(self, pattern: str, action_policy: str, confidence: int = 85) -> int:
        now = _now_iso()
        p_clean = pattern.strip()
        a_clean = action_policy.strip()
        conf = max(0, min(100, confidence))
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM brain_rules WHERE pattern = ?", (p_clean,)
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE brain_rules
                    SET action_policy = ?, confidence = MAX(confidence, ?), updated_at = ?
                    WHERE id = ?
                    """,
                    (a_clean, conf, now, existing["id"]),
                )
                return int(existing["id"])
            cur = conn.execute(
                """
                INSERT INTO brain_rules (pattern, action_policy, confidence, hits, created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?)
                """,
                (p_clean, a_clean, conf, now, now),
            )
            return int(cur.lastrowid)

    def list_brain_rules(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM brain_rules ORDER BY confidence DESC, hits DESC, id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def is_ignored_by_brain_rule(self, text: str, min_confidence: int = 80) -> bool:
        """Return True if text matches a high-confidence learned IGNORE rule."""
        low = (text or "").lower()
        if not low:
            return False
        stop = {
            "your", "with", "that", "this", "from", "have", "would", "like",
            "currently", "using", "files", "older", "than", "space", "should",
        }
        mind = None
        for r in self.list_brain_rules():
            policy = str(r.get("action_policy", "")).strip().lower()
            if not policy.startswith("ignore") or int(r.get("confidence", 0)) < min_confidence:
                continue
            pat = str(r.get("pattern", "")).lower()
            if pat and pat in low:
                return True
            words = {w.strip(".,?!:;()[]\"'") for w in pat.replace("_", " ").split()}
            words = {w for w in words if len(w) >= 4 and w not in stop}
            if len(words) >= 2 and sum(1 for w in words if w in low) >= 2:
                return True
            try:
                if pat:
                    if mind is None:
                        from .local_mind import LocalMind

                        mind = LocalMind(self)
                    if mind.semantic_similarity(pat, low, use_gpu_embed=False) >= 0.42:
                        return True
            except Exception:
                pass
        try:
            from .neural import NeuralEngine

            for hit in NeuralEngine(self).spreading_activation(low, top_k=2):
                if (
                    hit.get("kind") == "rule"
                    and float(hit.get("activation", 0.0) or 0.0) >= 0.62
                    and "ignore" in str(hit.get("text") or "").lower()
                ):
                    return True
        except Exception:
            pass
        return False

    def reinforce_brain_rule(self, rule_id: int, delta: int = 5) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE brain_rules
                SET confidence = MIN(100, MAX(0, confidence + ?)),
                    hits = hits + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (delta, _now_iso(), rule_id),
            )

    def match_fast_action_rule(
        self, text: str, min_confidence: int = 92, min_hits: int = 5
    ) -> dict[str, Any] | None:
        """Return a mature non-ignore rule matching text for zero-LLM fast-path execution."""
        low = (text or "").lower()
        if not low:
            return None
        for r in self.list_brain_rules():
            policy = str(r.get("action_policy", "")).strip()
            if not policy or policy.lower().startswith("ignore"):
                continue
            if int(r.get("confidence", 0)) < min_confidence or int(r.get("hits", 0)) < min_hits:
                continue
            pat = str(r.get("pattern", "")).lower()
            if pat and pat in low:
                return r
            words = {w.strip(".,?!:;()[]\"'") for w in pat.replace("_", " ").split() if len(w) >= 4}
            if len(words) >= 2 and sum(1 for w in words if w in low) >= 2:
                return r
        return None

    def delete_brain_rule(self, rule_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM brain_rules WHERE id = ?", (rule_id,))
            return cur.rowcount > 0

    def update_brain_rule(self, rule_id: int, pattern: str, action_policy: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE brain_rules
                SET pattern = ?, action_policy = ?, updated_at = ?
                WHERE id = ?
                """,
                (pattern.strip(), action_policy.strip(), _now_iso(), rule_id),
            )
            return cur.rowcount > 0

    def decay_stale_rules(self, stale_days: int = 14, delta: int = 2) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE brain_rules
                SET confidence = MAX(20, confidence - ?)
                WHERE julianday('now', 'localtime') - julianday(updated_at) >= ?
                  AND confidence > 20
                """,
                (delta, stale_days),
            )
            return cur.rowcount

    def add_brain_thought(
        self, decision: str, confidence: int, summary: str, action_taken: str | None = None
    ) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO brain_thoughts (decision, confidence, summary, action_taken, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (decision, confidence, summary, action_taken, _now_iso()),
            )
            self._fts_dirty = True
            return int(cur.lastrowid)

    def list_brain_thoughts(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM brain_thoughts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def create_brain_question(
        self, question: str, context: str, suggested_rule: str | None = None
    ) -> int:
        q_clean = question.strip()
        q_tokens = {w for w in q_clean.lower().split() if len(w) >= 3}
        with self._lock, self._connect() as conn:
            pending = conn.execute(
                "SELECT id, question, suggested_rule FROM brain_questions WHERE status = 'pending'"
            ).fetchall()
            for row in pending:
                if row["question"] == q_clean or (suggested_rule and row["suggested_rule"] == suggested_rule):
                    conn.execute(
                        "UPDATE brain_questions SET context = ? WHERE id = ?",
                        (context.strip(), row["id"]),
                    )
                    return int(row["id"])
                p_tokens = {w for w in str(row["question"]).lower().split() if len(w) >= 3}
                if q_tokens and p_tokens:
                    overlap = len(q_tokens & p_tokens) / max(1, len(q_tokens | p_tokens))
                    if overlap >= 0.6:
                        conn.execute(
                            "UPDATE brain_questions SET context = ? WHERE id = ?",
                            (context.strip(), row["id"]),
                        )
                        return int(row["id"])
            cur = conn.execute(
                """
                INSERT INTO brain_questions (question, context, suggested_rule, status, created_at)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (q_clean, context.strip(), suggested_rule, _now_iso()),
            )
            return int(cur.lastrowid)

    def expire_stale_brain_questions(self, max_age_hours: int = 48) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE brain_questions
                SET status = 'expired'
                WHERE status = 'pending'
                  AND (julianday('now', 'localtime') - julianday(created_at)) * 24.0 >= ?
                """,
                (float(max_age_hours),),
            )
            return cur.rowcount

    def get_brain_question(self, question_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM brain_questions WHERE id = ?", (question_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_brain_questions(self, status: str = "pending") -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM brain_questions WHERE status = ? ORDER BY id DESC",
                (status,),
            ).fetchall()
            return [dict(r) for r in rows]

    def resolve_brain_question(self, question_id: int, status: str, answer: str | None = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE brain_questions SET status = ?, user_answer = ? WHERE id = ?",
                (status, answer, question_id),
            )


    def add_evolve_proposal(
        self,
        file_path: str,
        category: str,
        title: str,
        reason: str,
        diff_text: str,
        new_content: str,
        test_passed: bool,
        status: str,
    ) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO evolve_proposals
                    (file_path, category, title, reason, diff_text, new_content, test_passed, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_path,
                    category,
                    title,
                    reason,
                    diff_text,
                    new_content,
                    1 if test_passed else 0,
                    status,
                    _now_iso(),
                ),
            )
            return int(cur.lastrowid)

    def list_evolve_proposals(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, file_path, category, title, reason, diff_text, test_passed, status, created_at "
                "FROM evolve_proposals ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_evolve_proposal(self, proposal_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evolve_proposals WHERE id = ?", (proposal_id,)
            ).fetchone()
            return dict(row) if row else None

    def update_evolve_status(self, proposal_id: int, status: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE evolve_proposals SET status = ? WHERE id = ?",
                (status, proposal_id),
            )


    def add_kg_edge(self, subject: str, relation: str, obj: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO kg_edges (subject, relation, object, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(subject, relation, object) DO UPDATE SET
                    updated_at = excluded.updated_at
                """,
                (subject.strip(), relation.strip(), obj.strip(), _now_iso()),
            )
            self._fts_dirty = True

    def query_kg(self, entity: str | None = None, limit: int = 40) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if entity and entity.strip():
                pat = f"%{entity.strip()}%"
                rows = conn.execute(
                    """
                    SELECT subject, relation, object, updated_at
                    FROM kg_edges
                    WHERE subject LIKE ? OR object LIKE ? OR relation LIKE ?
                    ORDER BY updated_at DESC LIMIT ?
                    """,
                    (pat, pat, pat, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT subject, relation, object, updated_at FROM kg_edges ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]


    def record_activity(self, category: str, detail: str) -> None:
        hour = datetime.now().hour
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO activity_log (hour_bucket, category, detail, created_at) VALUES (?, ?, ?, ?)",
                (hour, category, detail[:160], _now_iso()),
            )

    def get_recurring_habits(self, min_count: int = 3, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT hour_bucket, category, detail, COUNT(*) AS hits
                FROM activity_log
                GROUP BY hour_bucket, category, detail
                HAVING hits >= ?
                ORDER BY hits DESC
                LIMIT ?
                """,
                (min_count, limit),
            ).fetchall()
            return [dict(r) for r in rows]


    def add_lesson(self, context: str, insight: str, resolution: str) -> int:
        c_clean = context.strip()
        i_clean = insight.strip()
        r_clean = resolution.strip()
        now = _now_iso()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM lessons_learned WHERE context = ? AND insight = ?",
                (c_clean, i_clean),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE lessons_learned SET resolution = ?, created_at = ? WHERE id = ?",
                    (r_clean, now, existing["id"]),
                )
                self._fts_dirty = True
                return int(existing["id"])
            cur = conn.execute(
                "INSERT INTO lessons_learned (context, insight, resolution, created_at) VALUES (?, ?, ?, ?)",
                (c_clean, i_clean, r_clean, now),
            )
            self._fts_dirty = True
            return int(cur.lastrowid)

    def list_lessons(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM lessons_learned ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]


    def create_goal_plan(self, goal: str, steps_json: str) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO goal_plans (goal, steps_json, status, updated_at) VALUES (?, ?, 'active', ?)",
                (goal.strip(), steps_json, _now_iso()),
            )
            return int(cur.lastrowid)

    def update_goal_plan(self, plan_id: int, steps_json: str, status: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE goal_plans SET steps_json = ?, status = ?, updated_at = ? WHERE id = ?",
                (steps_json, status, _now_iso(), plan_id),
            )
            return cur.rowcount > 0

    def get_goal_plan(self, plan_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM goal_plans WHERE id = ?", (plan_id,)).fetchone()
            return dict(row) if row else None

    def list_goal_plans(self, limit: int = 15) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM goal_plans ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]


    def record_tool_metric(self, tool_name: str, elapsed_ms: float, is_error: bool = False) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_metrics (tool_name, calls, errors, total_ms, updated_at)
                VALUES (?, 1, ?, ?, ?)
                ON CONFLICT(tool_name) DO UPDATE SET
                    calls = calls + 1,
                    errors = errors + excluded.errors,
                    total_ms = total_ms + excluded.total_ms,
                    updated_at = excluded.updated_at
                """,
                (tool_name, 1 if is_error else 0, elapsed_ms, _now_iso()),
            )

    def get_tool_metrics(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT tool_name, calls, errors, ROUND(total_ms / MAX(1, calls), 1) AS avg_ms, updated_at
                FROM tool_metrics
                ORDER BY errors DESC, avg_ms DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def prune_database(self, keep_thoughts: int = 100, keep_activity_days: int = 30) -> dict[str, int]:
        deleted: dict[str, int] = {}
        with self._lock, self._connect() as conn:
            c1 = conn.execute(
                "DELETE FROM brain_thoughts WHERE id NOT IN (SELECT id FROM brain_thoughts ORDER BY id DESC LIMIT ?)",
                (keep_thoughts,),
            )
            deleted["thoughts"] = c1.rowcount
            c2 = conn.execute(
                "DELETE FROM activity_log WHERE julianday('now', 'localtime') - julianday(created_at) > ?",
                (keep_activity_days,),
            )
            deleted["activity_log"] = c2.rowcount
            c3 = conn.execute(
                "DELETE FROM brain_rules WHERE confidence <= 20 AND hits <= 1"
            )
            deleted["stale_rules"] = c3.rowcount
            c4 = conn.execute(
                "DELETE FROM evolve_proposals WHERE id NOT IN (SELECT id FROM evolve_proposals ORDER BY id DESC LIMIT 50)"
            )
            deleted["proposals"] = c4.rowcount
            c5 = conn.execute(
                "DELETE FROM model_quota WHERE julianday('now', 'localtime') - julianday(usage_date) > 30"
            )
            deleted["quota_rows"] = c5.rowcount
            c6 = conn.execute(
                "DELETE FROM audit_log WHERE id NOT IN (SELECT id FROM audit_log ORDER BY id DESC LIMIT 100)"
            )
            deleted["audit_log"] = c6.rowcount
            c7 = conn.execute("DELETE FROM focus_timers WHERE completed = 1")
            deleted["focus_timers"] = c7.rowcount
        with self._lock:
            conn = self._connect()
            conn.commit()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.execute("PRAGMA optimize;")
        return deleted


    def record_model_request(self, model: str) -> int:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_quota (usage_date, model, requests)
                VALUES (?, ?, 1)
                ON CONFLICT(usage_date, model) DO UPDATE SET
                    requests = requests + 1
                """,
                (today, model),
            )
            row = conn.execute(
                "SELECT requests FROM model_quota WHERE usage_date = ? AND model = ?",
                (today, model),
            ).fetchone()
            return int(row["requests"]) if row else 1

    def get_model_quota_today(self) -> dict[str, int]:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT model, requests FROM model_quota WHERE usage_date = ? ORDER BY requests DESC",
                (today,),
            ).fetchall()
            return {r["model"]: int(r["requests"]) for r in rows}


    def get_active_prompt(self, prompt_key: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prompt_versions WHERE prompt_key = ? AND active = 1 ORDER BY version DESC LIMIT 1",
                (prompt_key,),
            ).fetchone()
            return dict(row) if row else None

    def save_prompt_version(self, prompt_key: str, template: str, reason: str) -> int:
        now = _now_iso()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_v FROM prompt_versions WHERE prompt_key = ?",
                (prompt_key,),
            ).fetchone()
            next_ver = int(row["max_v"]) + 1
            conn.execute(
                "UPDATE prompt_versions SET active = 0 WHERE prompt_key = ?",
                (prompt_key,),
            )
            conn.execute(
                """
                INSERT INTO prompt_versions (prompt_key, version, template, reason, active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                """,
                (prompt_key, next_ver, template, reason.strip(), now),
            )
            return next_ver

    def rollback_prompt_version(self, prompt_key: str) -> int | None:
        with self._lock, self._connect() as conn:
            cur_row = conn.execute(
                "SELECT id, version FROM prompt_versions WHERE prompt_key = ? AND active = 1 ORDER BY version DESC LIMIT 1",
                (prompt_key,),
            ).fetchone()
            if not cur_row:
                return None
            conn.execute("UPDATE prompt_versions SET active = 0 WHERE id = ?", (cur_row["id"],))
            prev_row = conn.execute(
                "SELECT id, version FROM prompt_versions WHERE prompt_key = ? AND version < ? ORDER BY version DESC LIMIT 1",
                (prompt_key, cur_row["version"]),
            ).fetchone()
            if prev_row:
                conn.execute("UPDATE prompt_versions SET active = 1 WHERE id = ?", (prev_row["id"],))
                return int(prev_row["version"])
            return 0

    def list_prompt_versions(self, prompt_key: str | None = None) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if prompt_key:
                rows = conn.execute(
                    "SELECT * FROM prompt_versions WHERE prompt_key = ? ORDER BY version DESC",
                    (prompt_key,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM prompt_versions ORDER BY prompt_key ASC, version DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def record_prompt_outcome(self, prompt_key: str, success: bool) -> dict[str, Any] | None:
        """Increment successes or failures for the currently active custom prompt version."""
        col = "successes" if success else "failures"
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "SELECT id FROM prompt_versions WHERE prompt_key = ? AND active = 1 ORDER BY version DESC LIMIT 1",
                (prompt_key,),
            ).fetchone()
            if not cur:
                return None
            conn.execute(
                f"UPDATE prompt_versions SET {col} = {col} + 1 WHERE id = ?",
                (cur["id"],),
            )
            row = conn.execute(
                "SELECT * FROM prompt_versions WHERE id = ?", (cur["id"],)
            ).fetchone()
            return dict(row) if row else None

    def get_learned_cues(self) -> dict[str, list[str]]:
        """Return Brain-learned dynamic intent cues keyed by category."""
        import json

        cats = (
            "stall_reply",
            "stall_user",
            "correction",
            "action_req",
            "complex",
            "style_concise",
            "style_detailed",
            "style_bullets",
            "style_plain",
        )
        raw = self.get_memory("_internal:learned_cues", include_expired=True)
        out: dict[str, list[str]] = {c: [] for c in cats}
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    for c in cats:
                        vals = data.get(c)
                        if isinstance(vals, list):
                            out[c] = [str(v).strip().lower() for v in vals if str(v).strip()][:40]
            except Exception:
                pass
        return out

    def merge_learned_cues(self, new_cues: dict[str, list[str]]) -> dict[str, list[str]]:
        """Merge newly discovered/translated cues into persistent dynamic cue storage."""
        import json

        cur = self.get_learned_cues()
        for k, vals in (new_cues or {}).items():
            if k not in cur or not isinstance(vals, list):
                continue
            seen = set(cur[k])
            for v in vals:
                s = str(v).strip().lower()
                if s and len(s) >= 2 and s not in seen:
                    cur[k].append(s)
                    seen.add(s)
            cur[k] = cur[k][-40:]
        self.set_memory("_internal:learned_cues", json.dumps(cur, ensure_ascii=False))
        return cur

