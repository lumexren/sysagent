"""Integration tests for FastAPI endpoints in sysagent.server."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

import sysagent.server as srv
from sysagent.storage import Storage


def run_server_tests() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        db = Storage(tmp / "server_test.db")
        db.set_memory("srv_mem", "hello")
        db.upsert_session("s_srv", "int_srv")
        db.add_message("s_srv", "user", "check disk usage")
        db.add_lesson("ctx_srv", "insight_srv", "res_srv")
        qid = db.create_brain_question("Proceed?", "{}")
        db.add_brain_rule("pattern_srv", "policy_srv", confidence=85)

        orig_storage = srv.storage
        orig_brain_storage = srv.brain.storage
        orig_evolver_storage = srv.evolver.storage
        orig_self_storage = srv.self_monitor.storage
        srv.storage = db
        srv.brain.storage = db
        srv.evolver.storage = db
        srv.self_monitor.storage = db
        try:
            client = TestClient(srv.app)
            assert client.get("/").status_code == 200
            assert client.get("/api/i18n").status_code == 200
            assert client.get("/api/events").status_code == 200
            assert len(client.get("/api/sessions").json()["sessions"]) >= 1
            assert len(client.get("/api/sessions/s_srv/messages").json()["messages"]) == 1
            assert len(client.get("/api/memories").json()["memories"]) >= 1
            assert client.delete("/api/memories/srv_mem").status_code == 200
            assert client.get("/api/automations").status_code == 200
            assert client.get("/api/brain").status_code == 200
            assert len(client.get("/api/brain/questions").json()["questions"]) >= 1
            assert client.post(f"/api/brain/dismiss/{qid}").status_code == 200
            assert len(client.get("/api/brain/rules").json()["rules"]) >= 1
            assert client.get("/api/plugins").status_code == 200
            assert client.get("/api/evolve").status_code == 200
            assert client.get("/api/evolve/proposals").status_code == 200
            assert client.get("/api/processes").status_code == 200
            assert client.get("/api/logs").status_code == 200
            assert client.get("/api/self").status_code == 200
            assert client.get("/api/neural").status_code == 200
            assert client.post("/api/neural/pulse", json={"query": "ram status"}).status_code == 200
            assert "memories" in client.get("/api/self/export").json()
            assert client.get("/api/plans").status_code == 200
            assert len(client.get("/api/lessons").json()["lessons"]) >= 1
            assert client.get("/api/search", params={"q": "disk"}).status_code == 200
        finally:
            srv.storage = orig_storage
            srv.brain.storage = orig_brain_storage
            srv.evolver.storage = orig_evolver_storage
            srv.self_monitor.storage = orig_self_storage


if __name__ == "__main__":
    run_server_tests()
    print("SERVER_TESTS_OK")
