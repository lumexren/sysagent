![sysagent](https://img.shields.io/badge/sysagent-gray?style=for-the-badge&logo=python&logoColor=3776AB)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/lumexren/sysagent?style=for-the-badge&color=DAA520)
![Issues](https://img.shields.io/github/issues/lumexren/sysagent?style=for-the-badge)
![Last Commit](https://img.shields.io/github/last-commit/lumexren/sysagent?style=for-the-badge&color=F7A41D)
<br>

**Autonomous, self-evolving Linux system daemon and AI assistant powered by a hybrid Cloud + Local Neural architecture**

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Active Capabilities](#active-capabilities-auto-generated)

## Overview

`sysagent` runs continuously as a user service, monitoring system health, filesystems, processes, and systemd units while providing a Terminal CLI and UNIX pipe mode. It pairs cloud Gemini models with a local NumPy `NeuralEngine`, `CognitiveBrain`, `LocalMind`, and `SelfEvolver`.

## Features

- **Autonomous System Watcher** - Monitors CPU, RAM, disk burn rate, temperatures, GPU/VRAM, listening ports, USB devices, and watched directories with transient-spike filtering
- **Vectorized NeuralEngine** - 56K-parameter multi-head NumPy neural network (`256->128->128`) with AdamW, experience replay, Shannon-entropy uncertainty gating, spreading-activation associative memory, and a 4->2->4 telemetry anomaly autoencoder
- **CognitiveBrain** - Evaluates notable events, runs predictive pre-emption, validates autonomous actions via dry-run & post-action impact checks, and dynamically learns multilingual intent cues
- **SelfEvolver** - AST-guided code inspection, Ruff lint gate, isolated staging tests, A/B latency benchmarking, blast-radius guard, and automatic Git checkpoints
- **LocalMind & GGUF Cascade** - Local distillation, LoRA role adapters, shadow evaluation arena, and isolated `llama-server` subprocess execution with automatic idle RAM/VRAM cleanup
- **Smart Safety & Undo** - Auto-approves read-only tools, prompts on mutations with dry-run previews + Neural risk warnings, and records reversible file/config backups

## Architecture

| Component | Module | Responsibility |
|:----------|:-------|:---------------|
| **Agent** | [`agent.py`](sysagent/agent.py) | Multi-step tool loop, semantic safe-plan fast-path, self-supervised stall guard |
| **NeuralEngine** | [`neural.py`](sysagent/neural.py) | Tool/role routing, next-tool chain, world model forecast, anomaly autoencoder, associative synapses |
| **CognitiveBrain** | [`brain.py`](sysagent/brain.py) | Autonomous event triage, fast-path mature rules, dynamic cue learning, daily digest |
| **LocalMind** | [`local_mind.py`](sysagent/local_mind.py) | Online TF-IDF/centroid distillation, GGUF inference, shadow eval |
| **SelfEvolver** | [`evolve.py`](sysagent/evolve.py) | AST survey, isolated staging + Ruff gate, A/B benchmark, auto-rollback & git commit |
| **Watcher** | [`watcher.py`](sysagent/watcher.py) | Continuous telemetry, systemd/security monitoring, RSS leak guard |
| **Safety** | [`safety.py`](sysagent/safety.py) | Smart approval policy, dry-run impact preview, Neural risk warnings |
| **Storage** | [`storage.py`](sysagent/storage.py) | SQLite WAL + FTS5 full-text search, audit log, knowledge graph, learned cues |

## Installation

```bash
git clone https://github.com/lumexren/sysagent.git
cd sysagent
bash install.sh
```

Configure your credentials in `~/.config/sysagent/.env`, then enable and start the user daemon:

```bash
systemctl --user enable --now sysagent
```

## Configuration

Edit `~/.config/sysagent/.env`:

```ini
GEMINI_API_KEY=your_api_key_here
# Or use Antigravity CLI / OAuth / ADC:
# SYSAGENT_AUTH_MODE=agy_cli

# Optional: automatically push SelfEvolver commits to origin
# SYSAGENT_EVOLVE_AUTO_PUSH=true
```

## CLI Reference

| Command | Description |
|:--------|:------------|
| `sysagent chat` | Interactive terminal chat (`/model`, `/paste`, `/resume`, `/doctor`, `/digest`) |
| `sysagent status` | Rich grid overview of daemon, host/GPU, models/RPD, Brain, Evolve, and NeuralEngine |
| `sysagent doctor` | Diagnostics for daemon API, systemd unit, SQLite DB, NeuralEngine, GGUF models, disk, and Ruff |
| `sysagent digest` | Render the daily system and Brain digest |
| `sysagent neural [status\|train\|pulse]` | Inspect NeuralEngine metrics, retrain, or run a live query `pulse` |
| `sysagent brain [status\|think\|rules\|log]` | View Brain state, learned rules, and recent autonomous decisions |
| `sysagent evolve [status\|run\|log]` | View SelfEvolver state, stream a live scan, or list proposal history |
| `sysagent plugin [list\|enable\|disable\|remove]` | Manage user plugins in `~/.config/sysagent/plugins/` |
| `sysagent search "<query>"` | FTS5 full-text search across memories, lessons, and chat history |
| `cat err.log \| sysagent "fix"` | One-shot analysis from UNIX `stdin` pipes |

<!-- AUTO_CAPABILITIES_START -->
## Active Capabilities (Auto-Generated)

- `run_command`: Run a bash command on the user's Linux system. Use for inspecting processes, packages, services, files, or performing requested tasks.
- `read_file`: Read the contents of a text file from the local filesystem.
- `write_file`: Create or overwrite a text file on the local filesystem.
- `edit_file`: Replace an exact target snippet in an existing text file.
- `list_directory`: List files and folders inside a directory.
- `search_files`: Search for files by name pattern or grep text inside a directory.
- `system_status`: Get live CPU, RAM, disk usage, and boot time of the system.
- `list_events`: List recent background system events captured by the daemon listener.
- `send_notification`: Send a desktop notification to the user via notify-send.
- `remember_fact`: Save a persistent fact, preference, or rule to long-term memory.
- `recall_memories`: Retrieve all saved facts and preferences from long-term memory.
- `forget_memory`: Delete a saved fact from long-term memory by key.
- `create_automation`: Create a background automation rule. trigger_type can be: 'daily_time' (HH:MM, e.g. '09:00'), 'interval' (seconds, e.g. '3600'), or 'file_created' (filename glob in watched dirs, e.g. '*.pdf').
- `list_automations`: List all configured background automations.
- `delete_automation`: Delete an automation by its numeric ID.
- `read_clipboard`: Read the current text contents of the user's desktop clipboard.
- `take_screenshot`: Capture a screenshot of the user's screen so you can inspect it visually.
- `read_journal_logs`: Read recent systemd journalctl logs for a unit or the system.
- `desktop_control`: Control desktop hardware and state: volume_set (0-100), volume_mute (true/false), brightness_set (0-100), wifi (on/off), bluetooth (on/off), dnd (on/off), launch_app (app command/name).
- `list_audit_log`: List recent file edits/writes recorded for undo.
- `undo_last_action`: Undo the most recent file write/creation performed by the assistant.
- `start_focus_timer`: Start a focus/pomodoro timer that sends a desktop notification when finished.
- `get_app_usage`: Get tracked application/process usage time for today or a given YYYY-MM-DD date.
- `reload_plugins`: Reload custom Python plugins from ~/.config/sysagent/plugins/.
- `trash_file`: Safely move a file or directory to SysAgent's reversible trash instead of permanent deletion.
- `kg_add_relation`: Add a Knowledge Graph relation (subject -> relation -> object) to remember projects, paths, or entities.
- `kg_query`: Query the Knowledge Graph for relations matching an entity or keyword.
- `run_parallel_workers`: Run multiple independent read-only inspection/analysis subtasks in parallel using fast lightweight worker sub-agents. Use for multi-directory scans, batch analysis, or batch inspection.
- `forge_skill`: Create a new permanent Python plugin tool in ~/.config/sysagent/plugins/ and load it immediately. python_code must define TOOLS = [{'declaration': {'name': ..., 'description': ..., 'parameters': ...}, 'handler': fn, 'safe': bool}].
- `record_lesson`: Record an episodic lesson learned from a task failure or discovery so you never repeat the mistake.
- `create_goal_plan`: Create a structured multi-step plan for a complex user goal.
- `update_goal_step`: Update the status of a step in an active goal plan (0-based step_index).
- `web_search`: Search the internet for documentation, error solutions, or unknown facts.
- `fetch_url`: Fetch and extract readable text from a public HTTP/HTTPS URL.
- `search_memory_and_history`: Full-text search across past chat messages, memories, episodic lessons, knowledge graph, and autonomous brain thoughts using local FTS5 indexing.
- `undo_batch_actions`: Revert all file modifications and trashed files from the most recent multi-step turn/batch (or a specific batch_id) in reverse order.
- `list_processes`: List top running processes sorted by CPU or memory usage.
- `list_systemd_units`: List running or failed systemd service units.
- `inspect_self_component`: Inspect SysAgent's own internal components: 'brain' (recent thoughts, learned rules, pending questions), 'neural' (NeuralEngine weights, accuracy, synapses, world forecast, optional pulse), 'evolve' (self-evolution proposals & outcomes), 'local_mind' (shadow eval, coverage, roadmap), or 'all'.
<!-- AUTO_CAPABILITIES_END -->

---

<div align="center">

**Built with ❤️ by lumexren**

*Star ⭐ this repo if you find it useful!*

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

</div>
