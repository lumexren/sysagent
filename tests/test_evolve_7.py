import json
import tempfile
from pathlib import Path
from sysagent.tools import execute_tool

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "node_modules").mkdir()
    (root / "node_modules" / "ignored_target.txt").write_text("hello", encoding="utf-8")
    for i in range(60):
        (root / f"match_target_{i}.txt").write_text("hello world", encoding="utf-8")

    res = json.loads(execute_tool("search_files", {"query": "target", "path": str(root)}, None, None))
    assert len(res["matches"]) == 50
    assert all("node_modules" not in m for m in res["matches"])

    res_glob = json.loads(execute_tool("search_files", {"query": "*.txt", "path": str(root)}, None, None))
    assert len(res_glob["matches"]) == 50