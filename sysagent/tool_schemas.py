"""Tool JSON schema declarations exposed to the Gemini Interactions API."""

from __future__ import annotations

from typing import Any


TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "run_command",
        "description": (
            "Run a bash command on the user's Linux system. Use for inspecting processes, "
            "packages, services, files, or performing requested tasks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Optional working directory (defaults to user's home).",
                },
            },
            "required": ["command"],
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read the contents of a text file from the local filesystem.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "start_line": {
                    "type": "integer",
                    "description": "Optional 1-based start line.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional 1-based end line.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Create or overwrite a text file on the local filesystem.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Destination file path."},
                "content": {"type": "string", "description": "Full text content."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "type": "function",
        "name": "edit_file",
        "description": "Replace an exact target snippet in an existing text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Existing file path."},
                "target": {"type": "string", "description": "Exact text block to replace."},
                "replacement": {"type": "string", "description": "Replacement text block."},
            },
            "required": ["path", "target", "replacement"],
        },
    },
    {
        "type": "function",
        "name": "list_directory",
        "description": "List files and folders inside a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (supports ~).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "search_files",
        "description": "Search for files by name pattern or grep text inside a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Filename glob or text pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (defaults to ~).",
                },
                "content_search": {
                    "type": "boolean",
                    "description": "If true, search inside file contents; otherwise match filenames.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "system_status",
        "description": "Get live CPU, RAM, disk usage, and boot time of the system.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "function",
        "name": "list_events",
        "description": "List recent background system events captured by the daemon listener.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max number of events to return (default 20).",
                },
            },
        },
    },
    {
        "type": "function",
        "name": "send_notification",
        "description": "Send a desktop notification to the user via notify-send.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Notification title."},
                "body": {"type": "string", "description": "Notification body text."},
            },
            "required": ["title", "body"],
        },
    },
    {
        "type": "function",
        "name": "remember_fact",
        "description": "Save a persistent fact, preference, or rule to long-term memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short unique memory key (e.g. 'pdf_folder')."},
                "value": {"type": "string", "description": "Fact or preference to remember."},
                "expires_at": {"type": "string", "description": "Optional ISO expiration timestamp (e.g. '2026-12-01T00:00:00')."},
            },
            "required": ["key", "value"],
        },
    },
    {
        "type": "function",
        "name": "recall_memories",
        "description": "Retrieve all saved facts and preferences from long-term memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_expired": {"type": "boolean", "description": "Whether to include expired memories (default false)."},
            },
        },
    },
    {
        "type": "function",
        "name": "forget_memory",
        "description": "Delete a saved fact from long-term memory by key.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Memory key to remove."},
            },
            "required": ["key"],
        },
    },
    {
        "type": "function",
        "name": "create_automation",
        "description": (
            "Create a background automation rule. trigger_type can be: "
            "'daily_time' (HH:MM, e.g. '09:00'), "
            "'interval' (seconds, e.g. '3600'), or "
            "'file_created' (filename glob in watched dirs, e.g. '*.pdf')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short automation name."},
                "trigger_type": {
                    "type": "string",
                    "enum": ["daily_time", "interval", "file_created", "multi_condition"],
                },
                "trigger_value": {
                    "type": "string",
                    "description": "Trigger value (HH:MM, seconds, glob, or JSON for multi_condition e.g. '{\"max_cpu\":35,\"after_hour\":21,\"min_battery\":50}').",
                },
                "prompt": {"type": "string", "description": "Task prompt to run when triggered."},
            },
            "required": ["name", "trigger_type", "trigger_value", "prompt"],
        },
    },
    {
        "type": "function",
        "name": "list_automations",
        "description": "List all configured background automations.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "delete_automation",
        "description": "Delete an automation by its numeric ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "automation_id": {"type": "integer", "description": "Automation ID."},
            },
            "required": ["automation_id"],
        },
    },
    {
        "type": "function",
        "name": "read_clipboard",
        "description": "Read the current text contents of the user's desktop clipboard.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "take_screenshot",
        "description": "Capture a screenshot of the user's screen so you can inspect it visually.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "read_journal_logs",
        "description": "Read recent systemd journalctl logs for a unit or the system.",
        "parameters": {
            "type": "object",
            "properties": {
                "unit": {"type": "string", "description": "Optional systemd unit name (e.g. 'docker.service')."},
                "lines": {"type": "integer", "description": "Number of lines (default 40, max 200)."},
                "user_unit": {"type": "boolean", "description": "If true, query user systemd journal (--user)."},
            },
        },
    },
    {
        "type": "function",
        "name": "desktop_control",
        "description": (
            "Control desktop hardware and state: volume_set (0-100), volume_mute (true/false), "
            "brightness_set (0-100), wifi (on/off), bluetooth (on/off), dnd (on/off), launch_app (app command/name)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["volume_set", "volume_mute", "brightness_set", "wifi", "bluetooth", "dnd", "launch_app"],
                },
                "value": {"type": "string", "description": "Target value (e.g. '25', 'on', 'off', 'spotify')."},
            },
            "required": ["action", "value"],
        },
    },
    {
        "type": "function",
        "name": "list_audit_log",
        "description": "List recent file edits/writes recorded for undo.",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Max items (default 20)."}},
        },
    },
    {
        "type": "function",
        "name": "undo_last_action",
        "description": "Undo the most recent file write/creation performed by the assistant.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "start_focus_timer",
        "description": "Start a focus/pomodoro timer that sends a desktop notification when finished.",
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Duration in minutes (e.g. 25)."},
                "label": {"type": "string", "description": "Optional focus session label."},
            },
            "required": ["minutes"],
        },
    },
    {
        "type": "function",
        "name": "get_app_usage",
        "description": "Get tracked application/process usage time for today or a given YYYY-MM-DD date.",
        "parameters": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": "Optional YYYY-MM-DD date."}},
        },
    },
    {
        "type": "function",
        "name": "reload_plugins",
        "description": "Reload custom Python plugins from ~/.config/sysagent/plugins/.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "trash_file",
        "description": "Safely move a file or directory to SysAgent's reversible trash instead of permanent deletion.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to move to trash."}},
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "kg_add_relation",
        "description": "Add a Knowledge Graph relation (subject -> relation -> object) to remember projects, paths, or entities.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Entity name (e.g. 'garnet-oss')."},
                "relation": {"type": "string", "description": "Relationship (e.g. 'located_at', 'uses_language')."},
                "object": {"type": "string", "description": "Target value (e.g. '~/Documents/...')."},
            },
            "required": ["subject", "relation", "object"],
        },
    },
    {
        "type": "function",
        "name": "kg_query",
        "description": "Query the Knowledge Graph for relations matching an entity or keyword.",
        "parameters": {
            "type": "object",
            "properties": {"entity": {"type": "string", "description": "Optional entity or keyword filter."}},
        },
    },
    {
        "type": "function",
        "name": "run_parallel_workers",
        "description": (
            "Run multiple independent read-only inspection/analysis subtasks in parallel using fast lightweight worker sub-agents. "
            "Use for multi-directory scans, batch analysis, or batch inspection."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of independent subtask prompts to execute concurrently.",
                },
            },
            "required": ["tasks"],
        },
    },
    {
        "type": "function",
        "name": "forge_skill",
        "description": (
            "Create a new permanent Python plugin tool in ~/.config/sysagent/plugins/ and load it immediately. "
            "python_code must define TOOLS = [{'declaration': {'name': ..., 'description': ..., 'parameters': ...}, 'handler': fn, 'safe': bool}]."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "Snake-case filename/tool identifier (e.g. 'merge_pdfs')."},
                "python_code": {"type": "string", "description": "Complete Python module defining TOOLS."},
            },
            "required": ["tool_name", "python_code"],
        },
    },
    {
        "type": "function",
        "name": "record_lesson",
        "description": "Record an episodic lesson learned from a task failure or discovery so you never repeat the mistake.",
        "parameters": {
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "What task/command was attempted."},
                "insight": {"type": "string", "description": "What went wrong or was discovered."},
                "resolution": {"type": "string", "description": "What to do instead next time."},
            },
            "required": ["context", "insight", "resolution"],
        },
    },
    {
        "type": "function",
        "name": "create_goal_plan",
        "description": "Create a structured multi-step plan for a complex user goal.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Overall goal description."},
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of step titles.",
                },
            },
            "required": ["goal", "steps"],
        },
    },
    {
        "type": "function",
        "name": "update_goal_step",
        "description": "Update the status of a step in an active goal plan (0-based step_index).",
        "parameters": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "integer"},
                "step_index": {"type": "integer"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "done", "failed", "skipped"]},
                "note": {"type": "string"},
            },
            "required": ["plan_id", "step_index", "status"],
        },
    },
    {
        "type": "function",
        "name": "web_search",
        "description": "Search the internet for documentation, error solutions, or unknown facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "description": "Max results (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "fetch_url",
        "description": "Fetch and extract readable text from a public HTTP/HTTPS URL.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "HTTP or HTTPS URL."}},
            "required": ["url"],
        },
    },
    {
        "type": "function",
        "name": "search_memory_and_history",
        "description": (
            "Full-text search across past chat messages, memories, episodic lessons, knowledge graph, "
            "and autonomous brain thoughts using local FTS5 indexing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords or phrase to search in history and memory."},
                "limit": {"type": "integer", "description": "Max results (default 15)."},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "undo_batch_actions",
        "description": (
            "Revert all file modifications and trashed files from the most recent multi-step turn/batch "
            "(or a specific batch_id) in reverse order."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "batch_id": {"type": "string", "description": "Optional batch_id to revert (defaults to latest batch)."},
            },
        },
    },
    {
        "type": "function",
        "name": "list_processes",
        "description": "List top running processes sorted by CPU or memory usage.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max processes to return (default 15, max 50)."},
                "sort_by": {"type": "string", "description": "Sort key: 'cpu' (default) or 'memory'."},
            },
        },
    },
    {
        "type": "function",
        "name": "list_systemd_units",
        "description": "List running or failed systemd service units.",
        "parameters": {
            "type": "object",
            "properties": {
                "state": {"type": "string", "description": "Unit state filter: 'running' (default) or 'failed'."},
                "user_only": {"type": "boolean", "description": "Query user session units (--user)."},
                "limit": {"type": "integer", "description": "Max units to return (default 25)."},
            },
        },
    },
    {
        "type": "function",
        "name": "inspect_self_component",
        "description": (
            "Inspect SysAgent's live internal runtime components: 'prompts' (PromptRegistry active versions, "
            "prompt evolution history, win/loss metrics, reasons), 'brain' (recent thoughts, rules, questions), "
            "'cortex' (Flow Guardian, Dream Cycle, Predictive Healer, Habit Copilot), "
            "'neural' (NeuralEngine accuracy, world forecast, pulse), 'evolve' (proposals & outcomes), "
            "'local_mind' (warm GGUF model, backend, calibration, VRAM, vision readiness), "
            "'guard' (SelfGuard canary, eval gate, drift, circuit breakers, quota), 'notifications', "
            "'checkpoints' (cross-session semantic checkpoints), 'codebase' (live SysAgent Python modules & git commits), "
            "or 'all'. Call this for ANY question about SysAgent's own prompts, internal state, thoughts, rules, "
            "evolution, models, guard, checkpoints, or modules — regardless of the user's current working directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "component": {
                    "type": "string",
                    "description": (
                        "Target component: 'prompts', 'brain', 'cortex', 'neural', 'evolve', "
                        "'local_mind', 'guard', 'notifications', 'checkpoints', 'codebase', or 'all' (default 'all')."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": "Optional query text to run a live NeuralEngine pulse when inspecting 'neural'.",
                },
            },
        },
    },
    {
        "type": "function",
        "name": "switch_model",
        "description": (
            "Switch the active AI model for the current chat session (and optionally persist as default). "
            "Supported aliases: 'local' / 'local-mind' (local GGUF specialist ensemble), 'neon' (gemini-4-neon), "
            "'argon' (gemini-4-argon), 'hydrogen' (gemini-4-hydrogen), 'flash' (gemini-3.8-flash), 'lite' (gemini-3.5-flash-lite), or 'auto'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Target model alias or ID (e.g. 'local', 'neon', 'argon', 'hydrogen', 'flash', 'auto').",
                },
                "persist": {
                    "type": "boolean",
                    "description": "If true, also save as the default model in ~/.config/sysagent/.env.",
                },
            },
            "required": ["model"],
        },
    },
]
