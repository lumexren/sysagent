from sysagent.local_mind import LocalMindClient

client = LocalMindClient()

# 1. Test streaming when a tool call is predicted
tools = [{"name": "system_status"}]
stream_events = list(client.interactions.create(
    input="check system status cpu ram memory disk usage temperature",
    tools=tools,
    stream=True,
))
assert len(stream_events) >= 1
assert any(getattr(ev, "step", None) is not None or getattr(ev, "interaction", None) is not None for ev in stream_events)

# 2. Test streaming when input is a tool result list
res_events = list(client.interactions.create(
    input=[{"name": "system_status", "result": '{"cpu_percent": 12, "ram_percent": 40, "ram_used_gb": 4, "ram_total_gb": 16, "disk_percent": 50, "disk_free_gb": 100}'}],
    stream=True,
))
assert len(res_events) >= 1
texts = [ev.delta.text for ev in res_events if getattr(ev, "event_type", "") == "step.delta"]
assert any(("12%" in t or "%12" in t) and "cpu" in t.lower() for t in texts)
