# TARGETS_JSON: {"agent.py": [{"kind": "func", "sym": "build_system_instruction"}, {"cls": "AgentManager", "kind": "method", "sym": "resolve_approval"}, {"cls": "AgentManager", "kind": "method", "sym": "run_turn"}], "auth.py": [{"kind": "func", "sym": "build_genai_client"}, {"kind": "func", "sym": "get_auth_status"}, {"kind": "func", "sym": "logout_oauth"}, {"cls": "AuthManager", "kind": "method", "sym": "reload"}, {"kind": "func", "sym": "run_oauth_login"}], "brain.py": [{"cls": "CognitiveBrain", "kind": "method", "sym": "answer_question"}, {"cls": "CognitiveBrain", "kind": "method", "sym": "generate_daily_digest"}, {"cls": "CognitiveBrain", "kind": "method", "sym": "is_quiet_hours"}, {"cls": "CognitiveBrain", "kind": "method", "sym": "stop"}, {"cls": "CognitiveBrain", "kind": "method", "sym": "refresh_learned_cues"}], "config.py": [{"cls": "Settings", "kind": "attr", "sym": "base_url"}, {"kind": "func", "sym": "persist_env_var"}], "evolve.py": [{"cls": "SelfEvolver", "kind": "method", "sym": "apply_proposal"}, {"cls": "SelfEvolver", "kind": "method", "sym": "evolve_prompts_once"}, {"cls": "SelfEvolver", "kind": "method", "sym": "rollback_last_evolve"}], "evolve_ast.py": [{"kind": "func", "sym": "format_func_sig"}], "evolve_patch.py": [{"kind": "func", "sym": "apply_patch_text"}, {"kind": "func", "sym": "hot_reload_module"}], "evolve_safety.py": [{"kind": "func", "sym": "check_blast_radius"}], "i18n.py": [{"kind": "func", "sym": "get_translations"}, {"kind": "func", "sym": "resolve_locale"}], "local_mind.py": [{"cls": "LocalMind", "kind": "method", "sym": "distill_from_teachers"}, {"cls": "LocalMind", "kind": "method", "sym": "embed_text"}, {"kind": "func", "sym": "find_all_gguf_models"}, {"kind": "func", "sym": "find_gguf_model"}, {"cls": "LocalMind", "kind": "method", "sym": "penalize_tool"}, {"cls": "LocalMind", "kind": "method", "sym": "resolve_uncertain_batch"}, {"cls": "LocalMind", "kind": "method", "sym": "retrieve_similar_examples"}, {"cls": "LocalMind", "kind": "method", "sym": "run_gguf_inference"}, {"cls": "LocalMind", "kind": "method", "sym": "sample_user_utterances"}, {"cls": "LocalMind", "kind": "method", "sym": "synthesize_reply"}, {"cls": "LocalMind", "kind": "method", "sym": "try_local_cascade"}, {"kind": "func", "sym": "unload_idle_servers"}, {"cls": "LocalMind", "kind": "method", "sym": "weakest_specialist_role"}], "local_mind_client.py": [{"kind": "func", "sym": "download_default_gguf"}], "neural.py": [{"cls": "NeuralEngine", "kind": "method", "sym": "encode_features"}, {"cls": "NeuralEngine", "kind": "method", "sym": "predict_role_probs"}, {"cls": "NeuralEngine", "kind": "method", "sym": "predict_world_state"}], "plugins.py": [{"cls": "PluginRegistry", "kind": "method", "sym": "is_safe"}], "prompts.py": [{"cls": "PromptRegistry", "kind": "method", "sym": "get_template"}, {"cls": "PromptRegistry", "kind": "method", "sym": "list_status"}, {"cls": "PromptRegistry", "kind": "method", "sym": "render"}, {"cls": "PromptRegistry", "kind": "method", "sym": "rollback_prompt"}], "safety.py": [{"kind": "func", "sym": "compute_dry_run_preview"}, {"kind": "func", "sym": "neural_risk_note"}], "self_learn.py": [{"kind": "func", "sym": "cluster_signature"}, {"kind": "func", "sym": "create_git_checkpoint"}, {"kind": "func", "sym": "evolve_status_summary"}, {"kind": "func", "sym": "learn_status_summary"}, {"kind": "func", "sym": "run_local_self_play"}], "self_ops.py": [{"kind": "func", "sym": "choose_model"}, {"kind": "func", "sym": "compact_tool_schemas"}, {"kind": "func", "sym": "detect_style_feedback"}, {"kind": "func", "sym": "export_brain_bundle"}, {"cls": "SelfMonitor", "kind": "method", "sym": "get_self_health"}, {"kind": "func", "sym": "import_brain_bundle"}, {"cls": "ToolCache", "kind": "method", "sym": "invalidate_all"}, {"cls": "SelfMonitor", "kind": "method", "sym": "run_maintenance"}, {"kind": "func", "sym": "sync_capabilities_doc"}, {"kind": "func", "sym": "truncate_tool_output"}], "storage.py": [{"cls": "Storage", "kind": "method", "sym": "add_app_usage"}, {"cls": "Storage", "kind": "method", "sym": "create_automation"}, {"cls": "Storage", "kind": "method", "sym": "create_focus_timer"}, {"cls": "Storage", "kind": "method", "sym": "delete_automation"}, {"cls": "Storage", "kind": "method", "sym": "delete_memory"}, {"cls": "Storage", "kind": "method", "sym": "delete_session"}, {"cls": "Storage", "kind": "method", "sym": "get_app_usage"}, {"cls": "Storage", "kind": "method", "sym": "get_interaction_id"}, {"cls": "Storage", "kind": "method", "sym": "get_latest_undoable"}, {"cls": "Storage", "kind": "method", "sym": "get_messages"}, {"cls": "Storage", "kind": "method", "sym": "get_notification_cooldown_multiplier"}, {"cls": "Storage", "kind": "method", "sym": "get_recent_cross_session_messages"}, {"cls": "Storage", "kind": "method", "sym": "get_relevant_context"}, {"cls": "Storage", "kind": "method", "sym": "get_undoable_batch"}, {"cls": "Storage", "kind": "method", "sym": "list_audit_log"}, {"cls": "Storage", "kind": "method", "sym": "list_sessions"}, {"cls": "Storage", "kind": "method", "sym": "list_workflows"}, {"cls": "Storage", "kind": "method", "sym": "mark_automation_ran"}, {"cls": "Storage", "kind": "method", "sym": "mark_undone"}, {"cls": "Storage", "kind": "method", "sym": "pop_due_focus_timers"}, {"cls": "Storage", "kind": "method", "sym": "upsert_session"}, {"cls": "Storage", "kind": "method", "sym": "get_memory"}, {"cls": "Storage", "kind": "method", "sym": "prune_and_checkpoint"}, {"cls": "Storage", "kind": "method", "sym": "decay_memories"}, {"cls": "Storage", "kind": "method", "sym": "prune_expired_memories"}, {"cls": "Storage", "kind": "method", "sym": "resolve_memory_conflicts"}], "storage_brain.py": [{"cls": "_BrainStorageMixin", "kind": "method", "sym": "add_brain_rule"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "create_brain_question"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "decay_stale_rules"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "delete_brain_rule"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "get_active_prompt"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "get_brain_question"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "get_evolve_proposal"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "get_goal_plan"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "get_model_quota_today"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "get_recurring_habits"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "get_tool_metrics"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "is_ignored_by_brain_rule"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "list_brain_rules"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "list_brain_thoughts"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "list_goal_plans"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "list_lessons"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "list_prompt_versions"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "prune_database"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "query_kg"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "record_activity"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "record_model_request"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "reinforce_brain_rule"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "resolve_brain_question"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "rollback_prompt_version"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "update_brain_rule"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "update_evolve_status"}, {"cls": "_BrainStorageMixin", "kind": "method", "sym": "update_goal_plan"}], "tools.py": [{"kind": "func", "sym": "get_all_tool_declarations"}], "watcher.py": [{"cls": "SystemWatcher", "kind": "method", "sym": "add_event"}, {"cls": "SystemWatcher", "kind": "method", "sym": "get_events"}, {"cls": "SystemEvent", "kind": "method", "sym": "to_dict"}], "workers.py": [{"cls": "WorkerPool", "kind": "method", "sym": "run_parallel"}]}
"""Auto-synthesized signature and availability coverage tests."""

import importlib
import inspect
import os

os.environ.setdefault("SYSAGENT_BRAIN_ENABLED", "false")
os.environ.setdefault("SYSAGENT_EVOLVE_ENABLED", "false")

_mod_agent = importlib.import_module("sysagent.agent")
assert callable(_mod_agent.build_system_instruction) and inspect.signature(_mod_agent.build_system_instruction) is not None
assert callable(_mod_agent.AgentManager.resolve_approval) and inspect.signature(_mod_agent.AgentManager.resolve_approval) is not None
assert callable(_mod_agent.AgentManager.run_turn) and inspect.signature(_mod_agent.AgentManager.run_turn) is not None
_mod_auth = importlib.import_module("sysagent.auth")
assert callable(_mod_auth.build_genai_client) and inspect.signature(_mod_auth.build_genai_client) is not None
assert callable(_mod_auth.get_auth_status) and inspect.signature(_mod_auth.get_auth_status) is not None
assert callable(_mod_auth.logout_oauth) and inspect.signature(_mod_auth.logout_oauth) is not None
assert callable(_mod_auth.AuthManager.reload) and inspect.signature(_mod_auth.AuthManager.reload) is not None
assert callable(_mod_auth.run_oauth_login) and inspect.signature(_mod_auth.run_oauth_login) is not None
_mod_brain = importlib.import_module("sysagent.brain")
assert callable(_mod_brain.CognitiveBrain.answer_question) and inspect.signature(_mod_brain.CognitiveBrain.answer_question) is not None
assert callable(_mod_brain.CognitiveBrain.generate_daily_digest) and inspect.signature(_mod_brain.CognitiveBrain.generate_daily_digest) is not None
assert callable(_mod_brain.CognitiveBrain.is_quiet_hours) and inspect.signature(_mod_brain.CognitiveBrain.is_quiet_hours) is not None
assert callable(_mod_brain.CognitiveBrain.stop) and inspect.signature(_mod_brain.CognitiveBrain.stop) is not None
assert callable(_mod_brain.CognitiveBrain.refresh_learned_cues) and inspect.signature(_mod_brain.CognitiveBrain.refresh_learned_cues) is not None
_mod_config = importlib.import_module("sysagent.config")
assert hasattr(_mod_config.Settings, 'base_url')
assert callable(_mod_config.persist_env_var) and inspect.signature(_mod_config.persist_env_var) is not None
_mod_evolve = importlib.import_module("sysagent.evolve")
assert callable(_mod_evolve.SelfEvolver.apply_proposal) and inspect.signature(_mod_evolve.SelfEvolver.apply_proposal) is not None
assert callable(_mod_evolve.SelfEvolver.evolve_prompts_once) and inspect.signature(_mod_evolve.SelfEvolver.evolve_prompts_once) is not None
assert callable(_mod_evolve.SelfEvolver.rollback_last_evolve) and inspect.signature(_mod_evolve.SelfEvolver.rollback_last_evolve) is not None
_mod_evolve_ast = importlib.import_module("sysagent.evolve_ast")
assert callable(_mod_evolve_ast.format_func_sig) and inspect.signature(_mod_evolve_ast.format_func_sig) is not None
_mod_evolve_patch = importlib.import_module("sysagent.evolve_patch")
assert callable(_mod_evolve_patch.apply_patch_text) and inspect.signature(_mod_evolve_patch.apply_patch_text) is not None
assert callable(_mod_evolve_patch.hot_reload_module) and inspect.signature(_mod_evolve_patch.hot_reload_module) is not None
_mod_evolve_safety = importlib.import_module("sysagent.evolve_safety")
assert callable(_mod_evolve_safety.check_blast_radius) and inspect.signature(_mod_evolve_safety.check_blast_radius) is not None
_mod_i18n = importlib.import_module("sysagent.i18n")
assert callable(_mod_i18n.get_translations) and inspect.signature(_mod_i18n.get_translations) is not None
assert callable(_mod_i18n.resolve_locale) and inspect.signature(_mod_i18n.resolve_locale) is not None
_mod_local_mind = importlib.import_module("sysagent.local_mind")
assert callable(_mod_local_mind.LocalMind.distill_from_teachers) and inspect.signature(_mod_local_mind.LocalMind.distill_from_teachers) is not None
assert callable(_mod_local_mind.LocalMind.embed_text) and inspect.signature(_mod_local_mind.LocalMind.embed_text) is not None
assert callable(_mod_local_mind.find_all_gguf_models) and inspect.signature(_mod_local_mind.find_all_gguf_models) is not None
assert callable(_mod_local_mind.find_gguf_model) and inspect.signature(_mod_local_mind.find_gguf_model) is not None
assert callable(_mod_local_mind.LocalMind.penalize_tool) and inspect.signature(_mod_local_mind.LocalMind.penalize_tool) is not None
assert callable(_mod_local_mind.LocalMind.resolve_uncertain_batch) and inspect.signature(_mod_local_mind.LocalMind.resolve_uncertain_batch) is not None
assert callable(_mod_local_mind.LocalMind.retrieve_similar_examples) and inspect.signature(_mod_local_mind.LocalMind.retrieve_similar_examples) is not None
assert callable(_mod_local_mind.LocalMind.run_gguf_inference) and inspect.signature(_mod_local_mind.LocalMind.run_gguf_inference) is not None
assert callable(_mod_local_mind.LocalMind.sample_user_utterances) and inspect.signature(_mod_local_mind.LocalMind.sample_user_utterances) is not None
assert callable(_mod_local_mind.LocalMind.synthesize_reply) and inspect.signature(_mod_local_mind.LocalMind.synthesize_reply) is not None
assert callable(_mod_local_mind.LocalMind.try_local_cascade) and inspect.signature(_mod_local_mind.LocalMind.try_local_cascade) is not None
assert callable(_mod_local_mind.unload_idle_servers) and inspect.signature(_mod_local_mind.unload_idle_servers) is not None
assert callable(_mod_local_mind.LocalMind.weakest_specialist_role) and inspect.signature(_mod_local_mind.LocalMind.weakest_specialist_role) is not None
_mod_local_mind_client = importlib.import_module("sysagent.local_mind_client")
assert callable(_mod_local_mind_client.download_default_gguf) and inspect.signature(_mod_local_mind_client.download_default_gguf) is not None
_mod_neural = importlib.import_module("sysagent.neural")
assert callable(_mod_neural.NeuralEngine.encode_features) and inspect.signature(_mod_neural.NeuralEngine.encode_features) is not None
assert callable(_mod_neural.NeuralEngine.predict_role_probs) and inspect.signature(_mod_neural.NeuralEngine.predict_role_probs) is not None
assert callable(_mod_neural.NeuralEngine.predict_world_state) and inspect.signature(_mod_neural.NeuralEngine.predict_world_state) is not None
_mod_plugins = importlib.import_module("sysagent.plugins")
assert callable(_mod_plugins.PluginRegistry.is_safe) and inspect.signature(_mod_plugins.PluginRegistry.is_safe) is not None
_mod_prompts = importlib.import_module("sysagent.prompts")
assert callable(_mod_prompts.PromptRegistry.get_template) and inspect.signature(_mod_prompts.PromptRegistry.get_template) is not None
assert callable(_mod_prompts.PromptRegistry.list_status) and inspect.signature(_mod_prompts.PromptRegistry.list_status) is not None
assert callable(_mod_prompts.PromptRegistry.render) and inspect.signature(_mod_prompts.PromptRegistry.render) is not None
assert callable(_mod_prompts.PromptRegistry.rollback_prompt) and inspect.signature(_mod_prompts.PromptRegistry.rollback_prompt) is not None
_mod_safety = importlib.import_module("sysagent.safety")
assert callable(_mod_safety.compute_dry_run_preview) and inspect.signature(_mod_safety.compute_dry_run_preview) is not None
assert callable(_mod_safety.neural_risk_note) and inspect.signature(_mod_safety.neural_risk_note) is not None
_mod_self_learn = importlib.import_module("sysagent.self_learn")
assert callable(_mod_self_learn.cluster_signature) and inspect.signature(_mod_self_learn.cluster_signature) is not None
assert callable(_mod_self_learn.create_git_checkpoint) and inspect.signature(_mod_self_learn.create_git_checkpoint) is not None
assert callable(_mod_self_learn.evolve_status_summary) and inspect.signature(_mod_self_learn.evolve_status_summary) is not None
assert callable(_mod_self_learn.learn_status_summary) and inspect.signature(_mod_self_learn.learn_status_summary) is not None
assert callable(_mod_self_learn.run_local_self_play) and inspect.signature(_mod_self_learn.run_local_self_play) is not None
_mod_self_ops = importlib.import_module("sysagent.self_ops")
assert callable(_mod_self_ops.choose_model) and inspect.signature(_mod_self_ops.choose_model) is not None
assert callable(_mod_self_ops.compact_tool_schemas) and inspect.signature(_mod_self_ops.compact_tool_schemas) is not None
assert callable(_mod_self_ops.detect_style_feedback) and inspect.signature(_mod_self_ops.detect_style_feedback) is not None
assert callable(_mod_self_ops.export_brain_bundle) and inspect.signature(_mod_self_ops.export_brain_bundle) is not None
assert callable(_mod_self_ops.SelfMonitor.get_self_health) and inspect.signature(_mod_self_ops.SelfMonitor.get_self_health) is not None
assert callable(_mod_self_ops.import_brain_bundle) and inspect.signature(_mod_self_ops.import_brain_bundle) is not None
assert callable(_mod_self_ops.ToolCache.invalidate_all) and inspect.signature(_mod_self_ops.ToolCache.invalidate_all) is not None
assert callable(_mod_self_ops.SelfMonitor.run_maintenance) and inspect.signature(_mod_self_ops.SelfMonitor.run_maintenance) is not None
assert callable(_mod_self_ops.sync_capabilities_doc) and inspect.signature(_mod_self_ops.sync_capabilities_doc) is not None
assert callable(_mod_self_ops.truncate_tool_output) and inspect.signature(_mod_self_ops.truncate_tool_output) is not None
_mod_storage = importlib.import_module("sysagent.storage")
assert callable(_mod_storage.Storage.add_app_usage) and inspect.signature(_mod_storage.Storage.add_app_usage) is not None
assert callable(_mod_storage.Storage.create_automation) and inspect.signature(_mod_storage.Storage.create_automation) is not None
assert callable(_mod_storage.Storage.create_focus_timer) and inspect.signature(_mod_storage.Storage.create_focus_timer) is not None
assert callable(_mod_storage.Storage.delete_automation) and inspect.signature(_mod_storage.Storage.delete_automation) is not None
assert callable(_mod_storage.Storage.delete_memory) and inspect.signature(_mod_storage.Storage.delete_memory) is not None
assert callable(_mod_storage.Storage.delete_session) and inspect.signature(_mod_storage.Storage.delete_session) is not None
assert callable(_mod_storage.Storage.get_app_usage) and inspect.signature(_mod_storage.Storage.get_app_usage) is not None
assert callable(_mod_storage.Storage.get_interaction_id) and inspect.signature(_mod_storage.Storage.get_interaction_id) is not None
assert callable(_mod_storage.Storage.get_latest_undoable) and inspect.signature(_mod_storage.Storage.get_latest_undoable) is not None
assert callable(_mod_storage.Storage.get_messages) and inspect.signature(_mod_storage.Storage.get_messages) is not None
assert callable(_mod_storage.Storage.get_notification_cooldown_multiplier) and inspect.signature(_mod_storage.Storage.get_notification_cooldown_multiplier) is not None
assert callable(_mod_storage.Storage.get_recent_cross_session_messages) and inspect.signature(_mod_storage.Storage.get_recent_cross_session_messages) is not None
assert callable(_mod_storage.Storage.get_relevant_context) and inspect.signature(_mod_storage.Storage.get_relevant_context) is not None
assert callable(_mod_storage.Storage.get_undoable_batch) and inspect.signature(_mod_storage.Storage.get_undoable_batch) is not None
assert callable(_mod_storage.Storage.list_audit_log) and inspect.signature(_mod_storage.Storage.list_audit_log) is not None
assert callable(_mod_storage.Storage.list_sessions) and inspect.signature(_mod_storage.Storage.list_sessions) is not None
assert callable(_mod_storage.Storage.list_workflows) and inspect.signature(_mod_storage.Storage.list_workflows) is not None
assert callable(_mod_storage.Storage.mark_automation_ran) and inspect.signature(_mod_storage.Storage.mark_automation_ran) is not None
assert callable(_mod_storage.Storage.mark_undone) and inspect.signature(_mod_storage.Storage.mark_undone) is not None
assert callable(_mod_storage.Storage.pop_due_focus_timers) and inspect.signature(_mod_storage.Storage.pop_due_focus_timers) is not None
assert callable(_mod_storage.Storage.upsert_session) and inspect.signature(_mod_storage.Storage.upsert_session) is not None
assert callable(_mod_storage.Storage.get_memory) and inspect.signature(_mod_storage.Storage.get_memory) is not None
assert callable(_mod_storage.Storage.prune_and_checkpoint) and inspect.signature(_mod_storage.Storage.prune_and_checkpoint) is not None
assert callable(_mod_storage.Storage.decay_memories) and inspect.signature(_mod_storage.Storage.decay_memories) is not None
assert callable(_mod_storage.Storage.prune_expired_memories) and inspect.signature(_mod_storage.Storage.prune_expired_memories) is not None
assert callable(_mod_storage.Storage.resolve_memory_conflicts) and inspect.signature(_mod_storage.Storage.resolve_memory_conflicts) is not None
_mod_storage_brain = importlib.import_module("sysagent.storage_brain")
assert callable(_mod_storage_brain._BrainStorageMixin.add_brain_rule) and inspect.signature(_mod_storage_brain._BrainStorageMixin.add_brain_rule) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.create_brain_question) and inspect.signature(_mod_storage_brain._BrainStorageMixin.create_brain_question) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.decay_stale_rules) and inspect.signature(_mod_storage_brain._BrainStorageMixin.decay_stale_rules) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.delete_brain_rule) and inspect.signature(_mod_storage_brain._BrainStorageMixin.delete_brain_rule) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.get_active_prompt) and inspect.signature(_mod_storage_brain._BrainStorageMixin.get_active_prompt) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.get_brain_question) and inspect.signature(_mod_storage_brain._BrainStorageMixin.get_brain_question) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.get_evolve_proposal) and inspect.signature(_mod_storage_brain._BrainStorageMixin.get_evolve_proposal) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.get_goal_plan) and inspect.signature(_mod_storage_brain._BrainStorageMixin.get_goal_plan) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.get_model_quota_today) and inspect.signature(_mod_storage_brain._BrainStorageMixin.get_model_quota_today) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.get_recurring_habits) and inspect.signature(_mod_storage_brain._BrainStorageMixin.get_recurring_habits) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.get_tool_metrics) and inspect.signature(_mod_storage_brain._BrainStorageMixin.get_tool_metrics) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.is_ignored_by_brain_rule) and inspect.signature(_mod_storage_brain._BrainStorageMixin.is_ignored_by_brain_rule) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.list_brain_rules) and inspect.signature(_mod_storage_brain._BrainStorageMixin.list_brain_rules) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.list_brain_thoughts) and inspect.signature(_mod_storage_brain._BrainStorageMixin.list_brain_thoughts) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.list_goal_plans) and inspect.signature(_mod_storage_brain._BrainStorageMixin.list_goal_plans) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.list_lessons) and inspect.signature(_mod_storage_brain._BrainStorageMixin.list_lessons) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.list_prompt_versions) and inspect.signature(_mod_storage_brain._BrainStorageMixin.list_prompt_versions) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.prune_database) and inspect.signature(_mod_storage_brain._BrainStorageMixin.prune_database) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.query_kg) and inspect.signature(_mod_storage_brain._BrainStorageMixin.query_kg) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.record_activity) and inspect.signature(_mod_storage_brain._BrainStorageMixin.record_activity) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.record_model_request) and inspect.signature(_mod_storage_brain._BrainStorageMixin.record_model_request) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.reinforce_brain_rule) and inspect.signature(_mod_storage_brain._BrainStorageMixin.reinforce_brain_rule) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.resolve_brain_question) and inspect.signature(_mod_storage_brain._BrainStorageMixin.resolve_brain_question) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.rollback_prompt_version) and inspect.signature(_mod_storage_brain._BrainStorageMixin.rollback_prompt_version) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.update_brain_rule) and inspect.signature(_mod_storage_brain._BrainStorageMixin.update_brain_rule) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.update_evolve_status) and inspect.signature(_mod_storage_brain._BrainStorageMixin.update_evolve_status) is not None
assert callable(_mod_storage_brain._BrainStorageMixin.update_goal_plan) and inspect.signature(_mod_storage_brain._BrainStorageMixin.update_goal_plan) is not None
_mod_tools = importlib.import_module("sysagent.tools")
assert callable(_mod_tools.get_all_tool_declarations) and inspect.signature(_mod_tools.get_all_tool_declarations) is not None
_mod_watcher = importlib.import_module("sysagent.watcher")
assert callable(_mod_watcher.SystemWatcher.add_event) and inspect.signature(_mod_watcher.SystemWatcher.add_event) is not None
assert callable(_mod_watcher.SystemWatcher.get_events) and inspect.signature(_mod_watcher.SystemWatcher.get_events) is not None
assert callable(_mod_watcher.SystemEvent.to_dict) and inspect.signature(_mod_watcher.SystemEvent.to_dict) is not None
_mod_workers = importlib.import_module("sysagent.workers")
assert callable(_mod_workers.WorkerPool.run_parallel) and inspect.signature(_mod_workers.WorkerPool.run_parallel) is not None

print("AUTO_COVERAGE_TESTS_OK")
