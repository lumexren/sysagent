"""Lightweight i18n translations for SysAgent CLI and Web UI."""

from __future__ import annotations

import os
from typing import Any


TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "app_subtitle": "System Daemon & Assistant",
        "header_title": "Personal System Assistant",
        "header_status": "Active in background",
        "new_chat": "New Chat",
        "input_placeholder": "Ask anything... (e.g. summarize Downloads, which ports are open?)",
        "send": "Send",
        "system_status": "System Status",
        "name_label": "Name",
        "save": "Save",
        "locale_label": "Locale",
        "tab_events": "Events",
        "tab_sessions": "Chats",
        "tab_memories": "Memory",
        "tab_automations": "Automations",
        "tab_brain": "Brain",
        "tab_prompts": "Prompts",
        "tab_local": "Local AI",
        "undo_batch": "Undo Batch",
        "think_now": "Think Now",
        "answer_teach": "Answer & Teach",
        "dismiss": "Dismiss",
        "answer_placeholder": "Tell me what to do & remember...",
        "pending_questions": "Pending Decisions",
        "learned_rules": "Learned Rules",
        "thought_stream": "Thought Stream",
        "daily_digest": "Daily Digest",
        "edit": "Edit",
        "delete": "Delete",
        "scan_code": "Scan Code",
        "code_improvements": "Code Self-Improvements",
        "apply": "Apply",
        "reject": "Reject",
        "approval_required": "Approval Required",
        "approve": "Approve",
        "deny": "Deny",
        "chat_reset": "Chat reset.",
        "prev_task_running": "Previous task is still running.",
        "error_prefix": "Error",
        "cli_tool": "tool",
        "cli_confirm": "Allow this action? [y/N]: ",
        "cli_yes_tokens": "y,yes,e,evet",
        "cli_denied": "DENIED",
        "cli_ok": "OK",
        "cli_daemon_offline": (
            "SysAgent daemon is not running! Start it with:\n"
            "  systemctl --user start sysagent\n"
            "or in foreground:\n"
            "  {cli} serve"
        ),
        "cli_chat_started": "CLI chat started (session: {session_id}). Type 'exit' to quit.",
        "cli_you": "You",
        "cli_status_line": "Status       : {status} (Name: {name}, Model: {model}, Approval: {approval}, Locale: {locale})",
        "cli_api_key": "API Key      : {state}",
        "cli_configured": "Configured",
        "cli_missing": "MISSING!",
        "cli_cpu_ram": "CPU / RAM    : {cpu}% CPU | {ram}% RAM ({used}/{total} GB)",
        "cli_disk": "Disk (/)     : {disk}% used ({free} GB free)",
        "cli_web_ui": "Web UI       : {url}",
        "cli_no_events": "No background events recorded yet.",
        "cli_name_empty": "Name cannot be empty.",
        "cli_name_updated": "Assistant name updated: {name}",
        "cli_name_saved": "Assistant name saved ({path}): {name}",
        "quick_prompt": "What would you like me to do?",
        "quick_title": "{name} Quick",
        "cli_no_last_error": "No failed command recorded yet. Run `{cli} hook` to set up the shell hook.",
        "notif_yes": "Yes (Apply)",
        "notif_reply": "Reply...",
        "notif_dismiss": "Dismiss",
        "notif_approve": "Approve",
        "notif_deny": "Deny",
        "notif_inspect": "Inspect",
        "notif_open_ui": "Open Panel",
        "notif_action_done": "Action completed.",
        "notif_clean_cache": "Clean Cache",
        "notif_automate": "Yes, Automate",
        "notif_move_remember": "Move & Remember",
    },
    "tr": {
        "app_subtitle": "Sistem Daemon & Asistan",
        "header_title": "Kişisel Sistem Asistanı",
        "header_status": "Arka planda aktif",
        "new_chat": "Yeni Sohbet",
        "input_placeholder": "Bir şey iste... (ör: Downloads klasörünü özetle, hangi portlar açık?)",
        "send": "Gönder",
        "system_status": "Sistem Durumu",
        "name_label": "İsim",
        "save": "Kaydet",
        "locale_label": "Dil (Locale)",
        "tab_events": "Olaylar",
        "tab_sessions": "Sohbetler",
        "tab_memories": "Hafıza",
        "tab_automations": "Otomasyon",
        "tab_brain": "Beyin",
        "tab_prompts": "Promptlar",
        "tab_local": "Yerel AI",
        "undo_batch": "Toplu Geri Al",
        "think_now": "Şimdi Düşün",
        "answer_teach": "Yanıtla ve Öğret",
        "dismiss": "Yoksay",
        "answer_placeholder": "Ne yapmamı istediğini yaz (öğreneceğim)...",
        "pending_questions": "Bekleyen Kararlar",
        "learned_rules": "Öğrenilen Kurallar",
        "thought_stream": "Düşünce Günlüğü",
        "daily_digest": "Günlük Özet",
        "edit": "Düzenle",
        "delete": "Sil",
        "scan_code": "Kodu Tara",
        "code_improvements": "Kod İyileştirmeleri",
        "apply": "Uygula",
        "reject": "Reddet",
        "approval_required": "Onay Gerekli",
        "approve": "Onayla",
        "deny": "Reddet",
        "chat_reset": "Sohbet sıfırlandı.",
        "prev_task_running": "Önceki işlem henüz bitmedi.",
        "error_prefix": "Hata",
        "cli_tool": "araç",
        "cli_confirm": "İzin verilsin mi? [e/H]: ",
        "cli_yes_tokens": "e,evet,y,yes",
        "cli_denied": "REDDEDİLDİ",
        "cli_ok": "OK",
        "cli_daemon_offline": (
            "SysAgent daemon çalışmıyor! Başlatmak için:\n"
            "  systemctl --user start sysagent\n"
            "veya ön planda:\n"
            "  {cli} serve"
        ),
        "cli_chat_started": "CLI sohbeti başladı (oturum: {session_id}). Çıkmak için 'exit' yazın.",
        "cli_you": "Sen",
        "cli_status_line": "Durum        : {status} (İsim: {name}, Model: {model}, Onay: {approval}, Locale: {locale})",
        "cli_api_key": "API Anahtarı : {state}",
        "cli_configured": "Ayarlı",
        "cli_missing": "EKSİK!",
        "cli_cpu_ram": "CPU / RAM    : %{cpu} CPU | %{ram} RAM ({used}/{total} GB)",
        "cli_disk": "Disk (/)     : %{disk} dolu ({free} GB boş)",
        "cli_web_ui": "Web Arayüzü  : {url}",
        "cli_no_events": "Henüz kaydedilmiş sistem olayı yok.",
        "cli_name_empty": "İsim boş olamaz.",
        "cli_name_updated": "Asistanın adı güncellendi: {name}",
        "cli_name_saved": "Asistanın adı kaydedildi ({path}): {name}",
        "quick_prompt": "Ne yapmamı istersin?",
        "quick_title": "{name} Hızlı Komut",
        "cli_no_last_error": "Henüz kaydedilmiş hatalı komut yok. Shell kancasını kurmak için `{cli} hook` çalıştırın.",
        "notif_yes": "Evet (Uygula)",
        "notif_reply": "Yanıtla...",
        "notif_dismiss": "Yoksay",
        "notif_approve": "Onayla",
        "notif_deny": "Reddet",
        "notif_inspect": "İncele",
        "notif_open_ui": "Paneli Aç",
        "notif_action_done": "İşlem tamamlandı.",
        "notif_clean_cache": "Önbelleği Temizle",
        "notif_automate": "Evet, Otomatize Et",
        "notif_move_remember": "Taşı ve Öğren",
    },
}


def resolve_locale(locale: str | None) -> str:
    """Resolve a locale string (supports 'auto', 'tr-TR', 'en_US', etc.)."""
    loc = (locale or "auto").strip().lower()
    if loc == "auto":
        sys_lang = (os.environ.get("LC_ALL") or os.environ.get("LANG") or "en").lower()
        loc = sys_lang.split(".", 1)[0].split("_", 1)[0].split("-", 1)[0]
    else:
        loc = loc.split(".", 1)[0].split("_", 1)[0].split("-", 1)[0]
    return loc if loc in TRANSLATIONS else "en"


def get_translations(locale: str | None) -> dict[str, str]:
    """Return full translation dictionary for the resolved locale."""
    from .config import get_cli_bin_name

    cli_bin = get_cli_bin_name()
    return {
        k: (v.replace("{cli}", cli_bin) if "{cli}" in v else v)
        for k, v in TRANSLATIONS[resolve_locale(locale)].items()
    }


def t(key: str, locale: str | None = None, /, **kwargs: Any) -> str:
    """Translate a key for the given locale with optional format arguments (positional-only key/locale prevents kwargs collision)."""
    lang = resolve_locale(locale)
    template = TRANSLATIONS[lang].get(key) or TRANSLATIONS["en"].get(key, key)
    if "{cli}" in template and "cli" not in kwargs:
        from .config import get_cli_bin_name

        kwargs["cli"] = get_cli_bin_name()
    return template.format(**kwargs) if kwargs else template
