"""SQLite persistence for SysAgent: memories, chat history, and automations."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .storage_brain import _BrainStorageMixin


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_DB_LOCKS_GUARD = threading.Lock()
_DB_LOCKS: dict[str, threading.RLock] = {}


def _get_db_lock(db_path: Path) -> threading.RLock:
    key = str(db_path.resolve())
    with _DB_LOCKS_GUARD:
        lock = _DB_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _DB_LOCKS[key] = lock
        return lock


class Storage(_BrainStorageMixin):
    """Thread-safe SQLite storage for SysAgent."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (DATA_DIR / "sysagent.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _get_db_lock(self.db_path)
        self._conn: sqlite3.Connection | None = None
        self._fts_enabled = False
        self._fts_dirty = True
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=15000;")
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            self._conn = conn
        return self._conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    interaction_id TEXT,
                    title TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS automations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    trigger_value TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_run_at TEXT
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    target_path TEXT NOT NULL,
                    existed INTEGER NOT NULL,
                    backup_content TEXT,
                    batch_id TEXT,
                    created_at TEXT NOT NULL,
                    undone INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS app_usage (
                    usage_date TEXT NOT NULL,
                    app_name TEXT NOT NULL,
                    seconds INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (usage_date, app_name)
                );

                CREATE TABLE IF NOT EXISTS focus_timers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    ends_at TEXT NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS brain_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT NOT NULL,
                    action_policy TEXT NOT NULL,
                    confidence INTEGER NOT NULL DEFAULT 80,
                    hits INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS brain_thoughts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    action_taken TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS brain_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    context TEXT NOT NULL,
                    suggested_rule TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    user_answer TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evolve_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    diff_text TEXT NOT NULL,
                    new_content TEXT NOT NULL,
                    test_passed INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kg_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(subject, relation, object)
                );

                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hour_bucket INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lessons_learned (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context TEXT NOT NULL,
                    insight TEXT NOT NULL,
                    resolution TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS goal_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tool_metrics (
                    tool_name TEXT PRIMARY KEY,
                    calls INTEGER NOT NULL DEFAULT 0,
                    errors INTEGER NOT NULL DEFAULT 0,
                    total_ms REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_quota (
                    usage_date TEXT NOT NULL,
                    model TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (usage_date, model)
                );

                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_key TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    template TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    successes INTEGER NOT NULL DEFAULT 0,
                    failures INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS learned_workflows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signature TEXT UNIQUE NOT NULL,
                    goal_pattern TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS notification_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_key TEXT UNIQUE NOT NULL,
                    sample_text TEXT NOT NULL,
                    yes_count INTEGER NOT NULL DEFAULT 0,
                    dismiss_count INTEGER NOT NULL DEFAULT 0,
                    timeout_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_undo ON audit_log(undone, action, batch_id);
                CREATE INDEX IF NOT EXISTS idx_activity_bucket ON activity_log(hour_bucket, category);
                CREATE INDEX IF NOT EXISTS idx_proposals_status ON evolve_proposals(status, id DESC);
                CREATE INDEX IF NOT EXISTS idx_prompts_key_active ON prompt_versions(prompt_key, active, version DESC);
                """
            )
            for alter_sql in (
                "ALTER TABLE memories ADD COLUMN expires_at TEXT",
                "ALTER TABLE audit_log ADD COLUMN batch_id TEXT",
                "ALTER TABLE prompt_versions ADD COLUMN successes INTEGER NOT NULL DEFAULT 0",
                "ALTER TABLE prompt_versions ADD COLUMN failures INTEGER NOT NULL DEFAULT 0",
            ):
                try:
                    conn.execute(alter_sql)
                except sqlite3.OperationalError:
                    pass
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(source_type, source_id, title, body)"
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False


    def set_memory(self, key: str, value: str, expires_at: str | None = None) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (key, value, updated_at, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (key.strip(), value.strip(), _now_iso(), expires_at.strip() if expires_at else None),
            )
            self._fts_dirty = True

    def get_memory(self, key: str, include_expired: bool = True) -> str | None:
        """O(1) primary-key memory lookup."""
        now = _now_iso()
        with self._lock, self._connect() as conn:
            if include_expired:
                row = conn.execute("SELECT value FROM memories WHERE key = ?", (key.strip(),)).fetchone()
            else:
                row = conn.execute(
                    "SELECT value FROM memories WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
                    (key.strip(), now),
                ).fetchone()
            return str(row["value"]) if row else None

    def delete_memory(self, key: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM memories WHERE key = ?", (key.strip(),))
            if cur.rowcount > 0:
                self._fts_dirty = True
                return True
            return False

    def list_memories(
        self, include_expired: bool = False, include_internal: bool = True
    ) -> list[dict[str, Any]]:
        now = _now_iso()
        internal_clause = "" if include_internal else "key NOT LIKE '_internal:%'"
        with self._lock, self._connect() as conn:
            if include_expired:
                where = f"WHERE {internal_clause}" if internal_clause else ""
                rows = conn.execute(
                    f"SELECT key, value, updated_at, expires_at FROM memories {where} ORDER BY updated_at DESC"
                ).fetchall()
            else:
                exp_clause = "(expires_at IS NULL OR expires_at > ?)"
                where = f"WHERE {exp_clause}" + (f" AND {internal_clause}" if internal_clause else "")
                rows = conn.execute(
                    f"SELECT key, value, updated_at, expires_at FROM memories {where} ORDER BY updated_at DESC",
                    (now,),
                ).fetchall()
            return [dict(r) for r in rows]

    def prune_expired_memories(self) -> int:
        """Purge memories whose expires_at timestamp is in the past."""
        now = _now_iso()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            deleted = cur.rowcount
            if deleted > 0:
                self._fts_dirty = True
            return deleted

    def decay_memories(self, days_threshold: int = 60, prefix: str | None = None) -> int:
        """Remove old, non-internal memories that haven't been updated within days_threshold."""
        with self._lock, self._connect() as conn:
            query = """
                DELETE FROM memories
                WHERE key NOT LIKE '_internal:%'
                  AND julianday('now', 'localtime') - julianday(updated_at) >= ?
            """
            params: list[Any] = [float(days_threshold)]
            if prefix:
                query += " AND key LIKE ?"
                params.append(f"{prefix}%")
            cur = conn.execute(query, params)
            deleted = cur.rowcount
            if deleted > 0:
                self._fts_dirty = True
            return deleted

    def resolve_memory_conflicts(self, base_key: str, preferred_value: str | None = None) -> bool:
        """Resolve conflicting memory variations by key prefix, keeping either preferred or newest."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value, updated_at FROM memories WHERE key = ? OR key LIKE ? ORDER BY updated_at DESC",
                (base_key, f"{base_key}:%"),
            ).fetchall()
            if len(rows) <= 1:
                return False

            chosen_value = preferred_value if preferred_value is not None else str(rows[0]["value"])
            conn.execute(
                "DELETE FROM memories WHERE key = ? OR key LIKE ?",
                (base_key, f"{base_key}:%"),
            )
            conn.execute(
                "INSERT INTO memories (key, value, updated_at, expires_at) VALUES (?, ?, ?, NULL)",
                (base_key, chosen_value, _now_iso()),
            )
            self._fts_dirty = True
            return True


    def upsert_session(
        self, session_id: str, interaction_id: str | None, title: str | None = None
    ) -> None:
        now = _now_iso()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT title FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            final_title = (
                title
                if title
                else (existing["title"] if existing else f"Session {session_id[:8]}")
            )
            conn.execute(
                """
                INSERT INTO sessions (session_id, interaction_id, title, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    interaction_id = COALESCE(excluded.interaction_id, sessions.interaction_id),
                    updated_at = excluded.updated_at
                """,
                (session_id, interaction_id, final_title[:80], now),
            )

    def get_interaction_id(self, session_id: str) -> str | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT interaction_id FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return row["interaction_id"] if row else None

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, _now_iso()),
            )
            self._fts_dirty = True

    def get_messages(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def get_recent_cross_session_messages(
        self, exclude_session_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Return recent user/assistant messages across interactive sessions for cross-session continuity."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, role, content, created_at
                FROM messages
                WHERE session_id != COALESCE(?, '')
                  AND session_id NOT LIKE 'brain-%'
                  AND session_id NOT LIKE 'auto-%'
                  AND session_id NOT LIKE 'inspect-%'
                ORDER BY id DESC
                LIMIT ?
                """,
                (exclude_session_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def list_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, title, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


    def create_automation(
        self, name: str, trigger_type: str, trigger_value: str, prompt: str
    ) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO automations (name, trigger_type, trigger_value, prompt, enabled)
                VALUES (?, ?, ?, ?, 1)
                """,
                (name.strip(), trigger_type.strip(), trigger_value.strip(), prompt.strip()),
            )
            return int(cur.lastrowid)

    def list_automations(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM automations"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY id DESC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(query).fetchall()
            return [dict(r) for r in rows]

    def mark_automation_ran(self, automation_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE automations SET last_run_at = ? WHERE id = ?",
                (_now_iso(), automation_id),
            )

    def delete_automation(self, automation_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM automations WHERE id = ?", (automation_id,))
            return cur.rowcount > 0


    def record_file_backup(
        self,
        action: str,
        target_path: str,
        existed: bool,
        backup_content: str | None,
        batch_id: str | None = None,
    ) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO audit_log (action, target_path, existed, backup_content, batch_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (action, target_path, 1 if existed else 0, backup_content, batch_id, _now_iso()),
            )
            return int(cur.lastrowid)

    def get_latest_undoable(self, action: str | None = None) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            if action:
                row = conn.execute(
                    "SELECT * FROM audit_log WHERE undone = 0 AND action = ? ORDER BY id DESC LIMIT 1",
                    (action,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM audit_log WHERE undone = 0 AND action != 'evolve' ORDER BY id DESC LIMIT 1"
                ).fetchone()
            return dict(row) if row else None

    def get_undoable_batch(self, batch_id: str | None = None) -> list[dict[str, Any]]:
        """Return all undoable user audit entries belonging to a batch (or the latest batch/action)."""
        with self._lock, self._connect() as conn:
            target_batch = batch_id
            if not target_batch:
                latest = conn.execute(
                    "SELECT id, batch_id FROM audit_log WHERE undone = 0 AND action != 'evolve' ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if not latest:
                    return []
                target_batch = latest["batch_id"]
                if not target_batch:
                    row = conn.execute(
                        "SELECT * FROM audit_log WHERE id = ?", (latest["id"],)
                    ).fetchone()
                    return [dict(row)] if row else []
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE undone = 0 AND action != 'evolve' AND batch_id = ? ORDER BY id DESC",
                (target_batch,),
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_undone(self, audit_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE audit_log SET undone = 1 WHERE id = ?", (audit_id,))

    def list_audit_log(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, action, target_path, existed, batch_id, created_at, undone FROM audit_log ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]


    def add_app_usage(self, app_name: str, seconds: int) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_usage (usage_date, app_name, seconds)
                VALUES (?, ?, ?)
                ON CONFLICT(usage_date, app_name) DO UPDATE SET
                    seconds = seconds + excluded.seconds
                """,
                (today, app_name, seconds),
            )

    def get_app_usage(self, usage_date: str | None = None, limit: int = 15) -> list[dict[str, Any]]:
        target = usage_date or datetime.now().strftime("%Y-%m-%d")
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT app_name, seconds
                FROM app_usage
                WHERE usage_date = ?
                ORDER BY seconds DESC
                LIMIT ?
                """,
                (target, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def create_focus_timer(self, label: str, ends_at: str) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO focus_timers (label, ends_at, completed) VALUES (?, ?, 0)",
                (label, ends_at),
            )
            return int(cur.lastrowid)

    def pop_due_focus_timers(self) -> list[dict[str, Any]]:
        now = _now_iso()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, label, ends_at FROM focus_timers WHERE completed = 0 AND ends_at <= ?",
                (now,),
            ).fetchall()
            due = [dict(r) for r in rows]
            for item in due:
                conn.execute("UPDATE focus_timers SET completed = 1 WHERE id = ?", (item["id"],))
            return due


    def get_relevant_context(
        self,
        query: str,
        mem_limit: int = 15,
        kg_limit: int = 12,
        lesson_limit: int = 8,
    ) -> dict[str, list[dict[str, Any]]]:
        """Select memories, KG relations, and lessons most relevant to query (keeping style:* pinned)."""
        import re

        tokens = set(re.findall(r"\w{2,}", (query or "").lower()))
        q_grams = {
            w[i : i + 3] for w in tokens if len(w) >= 4 for i in range(len(w) - 2)
        }

        def _score(text: str, idx: int) -> tuple[float, int]:
            if not tokens:
                return (0.0, -idx)
            low = text.lower()
            hits = sum(1 for t in tokens if t in low)
            gram_hits = sum(0.25 for g in q_grams if g in low)
            return (hits + gram_hits, -idx)

        all_memories = [
            m for m in self.list_memories() if not str(m.get("key", "")).startswith("_internal:")
        ]
        pinned_style = [m for m in all_memories if str(m.get("key", "")).startswith("style:")]
        other_memories = [m for m in all_memories if not str(m.get("key", "")).startswith("style:")]
        scored_mem = sorted(
            enumerate(other_memories),
            key=lambda pair: _score(f"{pair[1]['key']} {pair[1]['value']}", pair[0]),
            reverse=True,
        )
        selected_memories = (pinned_style + [m for _, m in scored_mem])[:mem_limit]

        all_kg = self.query_kg(limit=100)
        scored_kg = sorted(
            enumerate(all_kg),
            key=lambda pair: _score(
                f"{pair[1]['subject']} {pair[1]['relation']} {pair[1]['object']}", pair[0]
            ),
            reverse=True,
        )
        selected_kg = [k for _, k in scored_kg[:kg_limit]]

        all_lessons = self.list_lessons(limit=60)
        scored_lessons = sorted(
            enumerate(all_lessons),
            key=lambda pair: _score(
                f"{pair[1]['context']} {pair[1]['insight']} {pair[1]['resolution']}", pair[0]
            ),
            reverse=True,
        )
        selected_lessons = [ls for _, ls in scored_lessons[:lesson_limit]]

        return {
            "memories": selected_memories,
            "kg_edges": selected_kg,
            "lessons": selected_lessons,
        }

    def search_knowledge(self, query: str, limit: int = 15) -> list[dict[str, Any]]:
        """Full-text search across memories, lessons, KG, brain thoughts, and chat messages using FTS5."""
        import re

        q_clean = (query or "").strip()
        if not q_clean:
            return []
        tokens = re.findall(r"\w{2,}", q_clean.lower())
        with self._lock, self._connect() as conn:
            if self._fts_enabled and tokens:
                try:
                    if self._fts_dirty:
                        conn.execute("DELETE FROM knowledge_fts")
                        conn.execute(
                            "INSERT INTO knowledge_fts(source_type, source_id, title, body) "
                            "SELECT 'memory', key, key, value FROM memories"
                        )
                        conn.execute(
                            "INSERT INTO knowledge_fts(source_type, source_id, title, body) "
                            "SELECT 'lesson', CAST(id AS TEXT), context, insight || ' -> ' || resolution FROM lessons_learned"
                        )
                        conn.execute(
                            "INSERT INTO knowledge_fts(source_type, source_id, title, body) "
                            "SELECT 'kg', CAST(id AS TEXT), subject || ' ' || relation, object FROM kg_edges"
                        )
                        conn.execute(
                            "INSERT INTO knowledge_fts(source_type, source_id, title, body) "
                            "SELECT 'thought', CAST(id AS TEXT), decision, summary || ' ' || COALESCE(action_taken, '') "
                            "FROM brain_thoughts ORDER BY id DESC LIMIT 100"
                        )
                        conn.execute(
                            "INSERT INTO knowledge_fts(source_type, source_id, title, body) "
                            "SELECT 'message', session_id, role, content FROM messages ORDER BY id DESC LIMIT 200"
                        )
                        self._fts_dirty = False
                    fts_query = " OR ".join(f'"{t}"*' for t in tokens[:8])
                    rows = conn.execute(
                        """
                        SELECT source_type, source_id, title, body, bm25(knowledge_fts) AS score
                        FROM knowledge_fts
                        WHERE knowledge_fts MATCH ?
                        ORDER BY score ASC
                        LIMIT ?
                        """,
                        (fts_query, limit),
                    ).fetchall()
                    if rows:
                        return [
                            {
                                "type": r["source_type"],
                                "id": r["source_id"],
                                "title": r["title"],
                                "detail": str(r["body"])[:300],
                            }
                            for r in rows
                        ]
                except sqlite3.OperationalError:
                    pass

            like_pat = f"%{q_clean}%"
            results: list[dict[str, Any]] = []
            for r in conn.execute(
                "SELECT key, value FROM memories WHERE key LIKE ? OR value LIKE ? LIMIT ?",
                (like_pat, like_pat, limit),
            ).fetchall():
                results.append({"type": "memory", "id": r["key"], "title": r["key"], "detail": r["value"]})
            for r in conn.execute(
                "SELECT id, context, insight, resolution FROM lessons_learned WHERE context LIKE ? OR insight LIKE ? OR resolution LIKE ? LIMIT ?",
                (like_pat, like_pat, like_pat, limit),
            ).fetchall():
                results.append(
                    {
                        "type": "lesson",
                        "id": str(r["id"]),
                        "title": r["context"],
                        "detail": f"{r['insight']} -> {r['resolution']}",
                    }
                )
            for r in conn.execute(
                "SELECT session_id, role, content FROM messages WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                (like_pat, limit),
            ).fetchall():
                results.append(
                    {
                        "type": "message",
                        "id": r["session_id"],
                        "title": f"{r['session_id']} ({r['role']})",
                        "detail": str(r["content"])[:300],
                    }
                )
            return results[:limit]


    def record_workflow(self, goal_pattern: str, tool_steps: list[dict[str, Any]]) -> int:
        """Record or reinforce a successful multi-step tool sequence as a reusable workflow."""
        import json as _json

        names = [str(s.get("name") or "").strip() for s in tool_steps if s.get("name")]
        if len(names) < 2:
            return 0
        sig = " -> ".join(names)
        goal_clean = (goal_pattern or "").strip()[:160]
        steps_json = _json.dumps(tool_steps[:8], ensure_ascii=False)
        now = _now_iso()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO learned_workflows (signature, goal_pattern, steps_json, hits, updated_at)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    goal_pattern = excluded.goal_pattern,
                    steps_json = excluded.steps_json,
                    hits = hits + 1,
                    updated_at = excluded.updated_at
                """,
                (sig, goal_clean, steps_json, now),
            )
            return int(cur.lastrowid)

    def find_matching_workflows(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        """Return learned workflows whose goal_pattern matches the user query."""
        import re

        tokens = set(re.findall(r"\w{2,}", (query or "").lower()))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, signature, goal_pattern, steps_json, hits, updated_at "
                "FROM learned_workflows ORDER BY hits DESC, updated_at DESC LIMIT 40"
            ).fetchall()
        scored: list[tuple[tuple[int, int], dict[str, Any]]] = []
        for r in rows:
            d = dict(r)
            low = f"{d['goal_pattern']} {d['signature']}".lower()
            overlap = sum(1 for t in tokens if t in low) if tokens else 0
            if overlap > 0:
                scored.append(((overlap, int(d["hits"])), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    def list_workflows(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM learned_workflows ORDER BY hits DESC, updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]


    def consolidate_memories(self) -> dict[str, int]:
        """Deduplicate normalized memories, brain rules, and lessons during maintenance."""
        mem_removed = 0
        rules_merged = 0
        lessons_merged = 0
        with self._lock, self._connect() as conn:
            m_rows = conn.execute(
                "SELECT key, value, updated_at FROM memories ORDER BY updated_at DESC"
            ).fetchall()
            seen_keys: set[str] = set()
            for r in m_rows:
                k = str(r["key"])
                v = str(r["value"] or "").strip()
                if k.startswith(("style:", "_internal:")):
                    continue
                norm = " ".join(k.lower().split())
                if len(v) < 2 or norm in seen_keys:
                    conn.execute("DELETE FROM memories WHERE key = ?", (k,))
                    mem_removed += 1
                else:
                    seen_keys.add(norm)

            r_rows = conn.execute(
                "SELECT id, pattern, confidence, hits FROM brain_rules ORDER BY confidence DESC, hits DESC, id DESC"
            ).fetchall()
            seen_patterns: set[str] = set()
            for r in r_rows:
                norm_p = " ".join(str(r["pattern"]).lower().split())
                if norm_p in seen_patterns:
                    conn.execute("DELETE FROM brain_rules WHERE id = ?", (r["id"],))
                    rules_merged += 1
                else:
                    seen_patterns.add(norm_p)

            l_rows = conn.execute(
                "SELECT id, context, insight FROM lessons_learned ORDER BY id DESC"
            ).fetchall()
            seen_lessons: set[tuple[str, str]] = set()
            for r in l_rows:
                pair = (
                    " ".join(str(r["context"]).lower().split()),
                    " ".join(str(r["insight"]).lower().split()),
                )
                if pair in seen_lessons:
                    conn.execute("DELETE FROM lessons_learned WHERE id = ?", (r["id"],))
                    lessons_merged += 1
                else:
                    seen_lessons.add(pair)
            if mem_removed or lessons_merged:
                self._fts_dirty = True

        return {
            "memories_removed": mem_removed,
            "rules_merged": rules_merged,
            "lessons_merged": lessons_merged,
        }


    @staticmethod
    def _normalize_feedback_key(text: str) -> str:
        import re

        raw = (text or "").strip()
        if raw.lower().startswith("approval:"):
            rest = raw[len("approval:") :].strip().lower()
            tokens = re.findall(r"[a-zA-Z0-9çğıöşüÇĞİÖŞÜ_.-]+", rest)
            suffix = " ".join(tokens[:6]) or rest[:50]
            return f"approval:{suffix}" if suffix else "approval:"
        tokens = [
            tok
            for tok in re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ_]{3,}", raw.lower())
            if tok not in {"the", "and", "for", "with", "this", "that", "would", "you", "like"}
        ]
        return " ".join(tokens[:6]) or raw.lower()[:60]

    def record_notification_feedback(self, pattern_text: str, outcome: str) -> dict[str, Any]:
        """Track user notification clicks (yes/dismiss/timeout) and auto-promote repeated habits."""
        pkey = self._normalize_feedback_key(pattern_text)
        if not pkey:
            return {"status": "ignored"}
        col_map = {"yes": "yes_count", "dismiss": "dismiss_count", "timeout": "timeout_count"}
        col = col_map.get(outcome)
        if not col:
            return {"status": "invalid_outcome"}
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO notification_feedback (pattern_key, sample_text, {col}, updated_at)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(pattern_key) DO UPDATE SET
                    {col} = {col} + 1,
                    sample_text = excluded.sample_text,
                    updated_at = excluded.updated_at
                """,
                (pkey, pattern_text[:200], now),
            )
            row = dict(
                conn.execute(
                    "SELECT * FROM notification_feedback WHERE pattern_key = ?", (pkey,)
                ).fetchone()
            )

        promoted: str | None = None
        if not pattern_text.startswith("approval:"):
            if outcome == "dismiss" and int(row["dismiss_count"]) >= 2:
                self.add_brain_rule(
                    pattern_text[:120],
                    "IGNORE silently: user repeatedly dismissed this notification",
                    confidence=92,
                )
                self.add_brain_thought(
                    "ignore",
                    92,
                    f"Auto-silenced notification pattern after {row['dismiss_count']}x dismiss: {pkey}",
                )
                promoted = "auto_ignore"
            elif outcome == "yes" and int(row["yes_count"]) >= 2:
                self.add_brain_rule(
                    pattern_text[:120],
                    "ACT autonomously without asking: user repeatedly approved this action",
                    confidence=95,
                )
                self.add_brain_thought(
                    "act",
                    95,
                    f"Auto-promoted to autonomous action after {row['yes_count']}x approval: {pkey}",
                )
                promoted = "auto_act"
        elif outcome == "yes" and int(row["yes_count"]) == 3:
            self.add_brain_thought(
                "act",
                95,
                f"Learned trusted approval habit (3x approved): {pattern_text}",
            )
            promoted = "auto_trust_approval"

        return {
            "pattern_key": pkey,
            "yes_count": int(row["yes_count"]),
            "dismiss_count": int(row["dismiss_count"]),
            "timeout_count": int(row["timeout_count"]),
            "promoted": promoted,
        }

    def is_approval_auto_trusted(self, pattern_text: str) -> bool:
        pkey = self._normalize_feedback_key(pattern_text)
        if not pkey:
            return False
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT yes_count, dismiss_count FROM notification_feedback WHERE pattern_key = ?",
                (pkey,),
            ).fetchone()
        if not row:
            return False
        return int(row["yes_count"]) >= 3 and int(row["dismiss_count"]) == 0

    def list_auto_trusted_approvals(self) -> list[str]:
        """Approval patterns that were approved >= 3 times and never dismissed (no longer prompted)."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT pattern_key FROM notification_feedback "
                "WHERE pattern_key LIKE 'approval:%' AND yes_count >= 3 AND dismiss_count = 0"
            ).fetchall()
        return [str(r["pattern_key"]) for r in rows]

    def get_notification_cooldown_multiplier(self, pattern_text: str) -> int:
        pkey = self._normalize_feedback_key(pattern_text)
        if not pkey:
            return 1
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT yes_count, timeout_count FROM notification_feedback WHERE pattern_key = ?",
                (pkey,),
            ).fetchone()
        if not row:
            return 1
        tc = int(row["timeout_count"])
        if tc >= 2 and int(row["yes_count"]) == 0:
            return min(4, 1 + tc // 2)
        return 1

    def prune_and_checkpoint(self, stale_days: int = 30) -> int:
        """Delete old routine ignore thoughts and checkpoint the SQLite WAL file safely."""
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    DELETE FROM brain_thoughts
                    WHERE decision = 'ignore'
                      AND julianday('now', 'localtime') - julianday(created_at) >= ?
                    """,
                    (float(stale_days),),
                )
                deleted = cur.rowcount
                if deleted > 0:
                    self._fts_dirty = True
            try:
                self._connect().execute("PRAGMA wal_checkpoint(PASSIVE);")
            except sqlite3.OperationalError:
                pass
            return deleted
