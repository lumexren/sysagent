import json, tempfile
from pathlib import Path
from unittest.mock import MagicMock
from sysagent.tools import execute_tool

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "sample.txt"
    p.write_text("\n".join(f"match_line_{i}" for i in range(100)), encoding="utf-8")
    res = json.loads(execute_tool("search_files", {"query": "match_line_", "path": td, "content_search": True}, MagicMock(), MagicMock()))
    assert len(res["matches"]) == 50, f"Expected 50 matches, got {len(res['matches'])}"