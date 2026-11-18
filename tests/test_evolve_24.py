import tempfile
from pathlib import Path
from sysagent.storage import Storage

with tempfile.TemporaryDirectory() as tmp:
    st = Storage(Path(tmp) / "test.db")
    for _ in range(3):
        res = st.record_notification_feedback("approval:run_command", "yes")
    assert res["promoted"] == "auto_trust_approval"
    assert st.is_approval_auto_trusted("approval:run_command") is True
    trusted = st.list_auto_trusted_approvals()
    assert "approval:run_command" in trusted, f"Expected approval:run_command in {trusted}"
