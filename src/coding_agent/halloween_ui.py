"""Halloween-themed Textual UI for the coding agent.

This module intentionally reuses ``coding_agent.agent.run_agent`` as the
single agent loop.  It observes the conversation list rather than parsing
stdout, and runs LLM-backed work in Textual background workers.
"""

'''THIS UI WAS CREATED FOR THIS CODING HARNESS, BY THE CODING HARNESS.'''

from __future__ import annotations

import asyncio
import inspect
import io
import json
import os
import threading
from contextlib import redirect_stdout
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, RichLog, Static

from coding_agent import llm, skills, tools
from coding_agent.agent import compact_history, run_agent

# ---------------------------------------------------------------------------
# Customization
# ---------------------------------------------------------------------------

APP_TITLE = "🎃 Haunted Guild Coding Agent"
APP_SUBTITLE = "All Hallows’ Eve edition · INIT AI Guild"

MIDNIGHT = "#0b0614"
PUMPKIN = "#ff9f1c"
TOXIC = "#8dff5a"
GHOST = "#f3efe0"
BLOOD = "#ff4d6d"
VIOLET = "#8a5cf6"

PREVIEW_LIMIT = 2200
TODO_PREVIEW_LIMIT = 240

WELCOME_TEXT = (
    "Welcome, wanderer. The veil is thin tonight.\n\n"
    "Ask the agent to inspect code, edit files, run tests, or summon a "
    "subagent. Use /help to see the old incantations.\n\n"
    "Ctrl+N starts a fresh conversation. Ctrl+Q flees the crypt. "
    "Escape returns you to the prompt."
)

HELP_TEXT = (
    "/help      show this help\n"
    "/new       start a fresh conversation\n"
    "/todos     show the current TODO list\n"
    "/skills    list available skills\n"
    "/model     show configured model information\n"
    "/compact   compact the conversation history\n"
    "/quit      leave the crypt\n"
    "/exit      leave the crypt\n\n"
    "Keys: Ctrl+N new · Ctrl+Q quit · Escape focus prompt"
)

TODO_STYLES = {
    "pending": GHOST,
    "in_progress": PUMPKIN,
    "done": TOXIC,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _resolve(value: Any) -> Any:
    return await value


def _invoke(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``func`` from a worker thread, supporting async callables."""

    result = func(*args, **kwargs)

    if inspect.isawaitable(result):
        result = asyncio.run(_resolve(result))

    return result


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    return {}


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            else:
                data = _as_dict(item)
                parts.append(
                    str(
                        data.get("text")
                        or data.get("content")
                        or item
                    )
                )
        return "\n".join(parts)
    return str(content)


def _clip(value: Any, limit: int = PREVIEW_LIMIT) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n… [truncated {len(text) - limit} chars]"


def _format_args(args: Any) -> str:
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return _clip(json.dumps(parsed, indent=2, ensure_ascii=False))
        except Exception:
            return _clip(args)

    data = _as_dict(args)
    if data:
        try:
            return _clip(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception:
            return _clip(str(data))

    if isinstance(args, (dict, list)):
        try:
            return _clip(json.dumps(args, indent=2, ensure_ascii=False))
        except Exception:
            pass

    return _clip(str(args))


def _usage_numbers(usage: Any) -> tuple[int, int, int]:
    data = _as_dict(usage)
    prompt = int(
        data.get("prompt_tokens")
        or data.get("input_tokens")
        or 0
    )
    completion = int(
        data.get("completion_tokens")
        or data.get("output_tokens")
        or 0
    )
    total = int(data.get("total_tokens") or (prompt + completion))
    return prompt, completion, total


def _model_name() -> str:
    for value in (
        getattr(llm, "MODEL", None),
        getattr(llm, "MODEL_NAME", None),
        os.environ.get("MODEL"),
    ):
        if value:
            return str(value)
    return "not configured"


def _conversation_cost() -> float:
    try:
        return float(getattr(llm, "CONVERSATION_COST", 0.0) or 0.0)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Observable message list
# ---------------------------------------------------------------------------


class ObservableMessages(list[dict[str, Any]]):
    """A message list that reports appends and slice replacements.

    ``run_agent`` mutates the supplied message list directly.  This wrapper
    lets the UI observe assistant/tool messages and compaction without
    changing the core agent.
    """

    def __init__(
        self,
        initial: list[dict[str, Any]],
        on_event: Any,
    ) -> None:
        super().__init__(initial)
        self._on_event = on_event
        self._suppress_depth = 0

    def _emit(self, event: dict[str, Any]) -> None:
        if self._suppress_depth:
            return
        try:
            self._on_event(event)
        except Exception:
            pass

    def suppress_events(self) -> None:
        self._suppress_depth += 1

    def unsuppress_events(self) -> None:
        self._suppress_depth = max(0, self._suppress_depth - 1)

    def append(self, item: dict[str, Any]) -> None:
        super().append(item)
        self._emit({"type": "message", "message": item})

    def extend(self, iterable: Any) -> None:
        items = list(iterable)
        super().extend(items)
        for item in items:
            self._emit({"type": "message", "message": item})

    def __iadd__(self, other: Any):
        self.extend(other)
        return self

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, slice):
            super().__setitem__(key, value)
            self._emit({"type": "compacted", "count": len(self)})
        else:
            super().__setitem__(key, value)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class HalloweenAgentApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        background: #0b0614;
        color: #f3efe0;
    }

    #title-bar {
        height: 3;
        content-align: center middle;
        text-style: bold;
        background: #2a0f3d;
        color: #ff9f1c;
        border-bottom: heavy #ff7a18;
    }

    #body {
        height: 1fr;
    }

    #main-column {
        width: 1fr;
        height: 1fr;
    }

    #conversation {
        width: 1fr;
        height: 1fr;
        background: #120a21;
        border: round #8a5cf6;
        padding: 0 1;
    }

    #sidebar {
        width: 42;
        height: 1fr;
        background: #160b27;
        border: round #8a5cf6;
        padding: 0 1;
        overflow-y: auto;
    }

    #todos-title, #usage-title, #keys-title {
        height: auto;
        margin-top: 1;
        color: #ff9f1c;
        text-style: bold;
    }

    #todos-text, #usage-text {
        height: auto;
        color: #f3efe0;
    }

    #keys-text {
        height: auto;
        color: #8dff5a;
    }

    #status-bar {
        height: 1;
        background: #2a0f3d;
        color: #8dff5a;
        padding: 0 1;
        content-align: left middle;
    }

    #prompt {
        height: 3;
        background: #1a0d2b;
        color: #f3efe0;
        border: tall #ff9f1c;
    }

    #prompt:focus {
        border: tall #8dff5a;
    }
    """

    BINDINGS = [
        Binding("ctrl+n", "new_conversation", "New conversation"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("escape", "focus_prompt", "Focus prompt", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._ui_thread_id = threading.get_ident()
        self._busy = False
        self._busy_text = ""
        self._assistant_messages_seen = 0
        self._tool_calls: dict[str, tuple[str, Any]] = {}

        self.messages = ObservableMessages(
            [{"role": "system", "content": llm.SYSTEM_PROMPT}],
            self._on_message_observed,
        )

        self.last_usage: tuple[int, int, int] = (0, 0, 0)
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.last_turn_cost = 0.0

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(f"{APP_TITLE}\n{APP_SUBTITLE}", id="title-bar")

        with Horizontal(id="body"):
            with Vertical(id="main-column"):
                yield RichLog(
                    id="conversation",
                    wrap=True,
                    markup=True,
                    auto_scroll=True,
                )

            with VerticalScroll(id="sidebar"):
                yield Static("📜 Grimoire of TODOs", id="todos-title")
                yield Static(id="todos-text")

                yield Static("🕯️ Candle Budget", id="usage-title")
                yield Static(id="usage-text")

                yield Static("🦇 Keys", id="keys-title")
                yield Static(
                    "Ctrl+N new · Ctrl+Q quit\nEsc focus prompt",
                    id="keys-text",
                )

        yield Static(id="status-bar")
        yield Input(
            placeholder="Speak into the dark… (/help for commands)",
            id="prompt",
        )

    def on_mount(self) -> None:
        self._ui_thread_id = threading.get_ident()
        self.title = APP_TITLE
        self.sub_title = APP_SUBTITLE

        conversation = self.query_one("#conversation", RichLog)
        conversation.border_title = "🕯️ Conversation"

        self._update_todos()
        self._update_usage()
        self._update_status()
        self._write_welcome()
        self.query_one("#prompt", Input).focus()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _write_panel(
        self,
        title: str,
        body: str | Text,
        border_style: str = VIOLET,
    ) -> None:
        if isinstance(body, str):
            body = Text(body)
        panel = Panel(
            body,
            title=Text(title),
            border_style=border_style,
            padding=(0, 1),
            expand=True,
        )
        self.query_one("#conversation", RichLog).write(panel)

    def _write_system(
        self,
        title: str,
        body: str | Text,
        border_style: str = TOXIC,
    ) -> None:
        self._write_panel(title, body, border_style)

    def _write_welcome(self) -> None:
        self._write_panel(
            "🕸️ The Haunted Guild",
            Text(WELCOME_TEXT, style=GHOST),
            PUMPKIN,
        )

    def _on_message_observed(self, event: dict[str, Any]) -> None:
        if threading.get_ident() == self._ui_thread_id:
            self._handle_observed_event(event)
        else:
            self._post_to_ui(self._handle_observed_event, event)

    def _handle_observed_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type == "compacted":
            count = event.get("count", len(self.messages))
            self._write_system(
                "🧹 Compaction",
                f"The conversation was compacted. {count} messages remain.",
                TOXIC,
            )
            return

        if event_type == "message":
            message = event.get("message")
            if isinstance(message, dict):
                self._render_message(message)

    def _render_message(self, message: dict[str, Any]) -> None:
        role = message.get("role")

        if role == "user":
            body = _clip(_content_to_text(message.get("content")))
            self._write_panel("🧙 You", body or "(empty message)", PUMPKIN)
            return

        if role == "assistant":
            self._assistant_messages_seen += 1
            content = _content_to_text(message.get("content"))
            if content.strip():
                self._write_panel("👻 Agent", _clip(content), GHOST)
            self._render_tool_calls(message)
            return

        if role == "tool":
            self._render_tool_result(message)
            self._update_todos()
            return

    def _render_tool_calls(self, message: dict[str, Any]) -> None:
        tool_calls = message.get("tool_calls") or []
        for tool_call in tool_calls:
            data = _as_dict(tool_call)
            function = _as_dict(data.get("function"))

            name = str(
                function.get("name")
                or data.get("name")
                or "tool"
            )
            args = function.get(
                "arguments",
                data.get("arguments", ""),
            )
            call_id = str(data.get("id") or "")

            if call_id:
                self._tool_calls[call_id] = (name, args)

            self._write_panel(
                f"🔮 Tool call · {name}",
                _format_args(args),
                TOXIC,
            )

    def _render_tool_result(self, message: dict[str, Any]) -> None:
        call_id = str(message.get("tool_call_id") or "")
        name, _args = self._tool_calls.get(
            call_id,
            (str(message.get("name") or "tool"), ""),
        )
        content = _content_to_text(message.get("content"))
        self._write_panel(
            f"🛠️ Tool result · {name}",
            _clip(content) or "(no output)",
            VIOLET,
        )

    # ------------------------------------------------------------------
    # Sidebar / status
    # ------------------------------------------------------------------

    def _format_todos_text(self) -> Text:
        text = Text()
        todos = getattr(tools, "TODOS", []) or []
        marks = getattr(tools, "MARKS", {}) or {}

        if not todos:
            text.append(
                "No TODOs yet.\nThe agent can create them with write_todos.",
                style="dim italic #b8a9c9",
            )
            return text

        for todo in todos:
            if isinstance(todo, dict):
                content = (
                    todo.get("content")
                    or todo.get("task")
                    or todo.get("title")
                    or json.dumps(todo, ensure_ascii=False)
                )
                status = str(todo.get("status") or "pending")
            else:
                content = str(todo)
                status = "pending"

            mark = str(marks.get(status, "•"))
            style = TODO_STYLES.get(status, GHOST)

            text.append(f"{mark} ", style=f"bold {style}")
            text.append(
                _clip(str(content), TODO_PREVIEW_LIMIT),
                style=style,
            )
            text.append("\n")

        return text

    def _update_todos(self) -> None:
        self.query_one("#todos-text", Static).update(self._format_todos_text())

    def _update_usage(self) -> None:
        text = Text()

        last_prompt, last_completion, last_total = self.last_usage
        session_total = (
            self.session_prompt_tokens + self.session_completion_tokens
        )
        cost = _conversation_cost()
        model = _model_name()

        text.append("Last turn\n", style=f"bold {PUMPKIN}")
        text.append(
            f"↑ {last_prompt:,}  ↓ {last_completion:,}  Σ {last_total:,}\n",
            style=GHOST,
        )

        text.append("Session\n", style=f"bold {PUMPKIN}")
        text.append(
            f"↑ {self.session_prompt_tokens:,}  "
            f"↓ {self.session_completion_tokens:,}  "
            f"Σ {session_total:,}\n",
            style=GHOST,
        )

        text.append("Cost\n", style=f"bold {PUMPKIN}")
        text.append(f"Session ${cost:.6f}\n", style=TOXIC)
        text.append(f"Last turn ${self.last_turn_cost:.6f}\n", style=GHOST)

        text.append("Model\n", style=f"bold {PUMPKIN}")
        text.append(_clip(model, 36), style=GHOST)

        self.query_one("#usage-text", Static).update(text)

    def _update_status(self) -> None:
        if self._busy:
            message = f"🕸️ {self._busy_text or 'Working'}… the candles flicker"
            style = PUMPKIN
        else:
            model = _model_name()
            message = (
                f"🕯️ Ready · {model} · /help · "
                "Ctrl+N new · Ctrl+Q quit"
            )
            style = TOXIC

        self.query_one("#status-bar", Static).update(
            Text(message, style=style)
        )

    def _set_busy(self, busy: bool, busy_text: str = "") -> None:
        self._busy = busy
        self._busy_text = busy_text

        prompt = self.query_one("#prompt", Input)
        prompt.disabled = busy
        if not busy:
            prompt.focus()

        self._update_status()

    def _post_to_ui(self, callback: Any, *args: Any) -> None:
        try:
            self.call_from_thread(callback, *args)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Input and slash commands
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""

        if not text:
            return

        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        if self._busy:
            self._write_system(
                "🕸️ Busy",
                "The agent is still working. Wait for the candles to settle.",
                BLOOD,
            )
            return

        self._start_user_turn(text)

    def _start_user_turn(self, text: str) -> None:
        self._assistant_messages_seen = 0
        self.messages.append({"role": "user", "content": text})
        self._set_busy(True, "Summoning the agent")
        self._agent_turn()

    def _handle_slash_command(self, text: str) -> None:
        command, _, _args = text.partition(" ")
        command = command.lower()

        if command in ("/quit", "/exit"):
            self.exit()
            return

        if command == "/help":
            self._write_system(
                "📖 Incantations",
                HELP_TEXT,
                PUMPKIN,
            )
            return

        if command == "/new":
            if self._busy:
                self._write_system(
                    "🕸️ Busy",
                    "Cannot start a new conversation while the agent is working.",
                    BLOOD,
                )
            else:
                self._new_conversation()
            return

        if command == "/todos":
            self._write_system(
                "📜 TODOs",
                self._format_todos_text(),
                PUMPKIN,
            )
            return

        if command == "/skills":
            self._show_skills()
            return

        if command == "/model":
            self._show_model()
            return

        if command == "/compact":
            if self._busy:
                self._write_system(
                    "🕸️ Busy",
                    "Cannot compact while the agent is working.",
                    BLOOD,
                )
            else:
                self._start_compact()
            return

        self._write_system(
            "🕯️ Unknown incantation",
            f"Unknown command: {command}\nTry /help.",
            BLOOD,
        )

    def _show_skills(self) -> None:
        try:
            result = skills.list_skills()
        except Exception as exc:
            self._write_system(
                "💀 Skill error",
                f"{type(exc).__name__}: {exc}",
                BLOOD,
            )
            return

        if isinstance(result, list):
            lines: list[str] = []
            for item in result:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("id") or item
                    description = item.get("description") or ""
                    if description:
                        lines.append(f"{name} — {description}")
                    else:
                        lines.append(str(name))
                else:
                    lines.append(str(item))
            body = "\n".join(lines) if lines else "No skills found."
        else:
            body = str(result)

        self._write_system("📚 Skills", body, TOXIC)

    def _show_model(self) -> None:
        body = f"Model: {_model_name()}"

        base_url = (
            getattr(llm, "BASE_URL", None)
            or os.environ.get("BASE_URL")
        )
        if base_url:
            body += f"\nBase URL: {base_url}"

        self._write_system("🔮 Model", body, VIOLET)

    def _new_conversation(self) -> None:
        if self._busy:
            return

        self.messages.suppress_events()
        try:
            self.messages[:] = [
                {"role": "system", "content": llm.SYSTEM_PROMPT}
            ]
        finally:
            self.messages.unsuppress_events()

        self._tool_calls.clear()

        try:
            tools.TODOS.clear()
        except Exception:
            pass

        llm.CONVERSATION_COST = 0.0

        self._assistant_messages_seen = 0
        self.last_usage = (0, 0, 0)
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.last_turn_cost = 0.0

        self.query_one("#conversation", RichLog).clear()
        self._write_welcome()
        self._update_todos()
        self._update_usage()
        self._set_busy(False)

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    @work(exclusive=True, thread=True)
    def _agent_turn(self) -> None:
        cost_before = _conversation_cost()
        content = ""
        usage: Any = None
        error: str | None = None
        buffer = io.StringIO()

        try:
            with redirect_stdout(buffer):
                content, usage = _invoke(run_agent, self.messages)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        cost_delta = max(0.0, _conversation_cost() - cost_before)
        self._post_to_ui(
            self._turn_finished,
            content,
            usage,
            cost_delta,
            error,
        )

    def _turn_finished(
        self,
        content: Any,
        usage: Any,
        cost_delta: float,
        error: str | None,
    ) -> None:
        self.last_turn_cost = float(cost_delta or 0.0)

        if usage is not None:
            prompt, completion, _total = _usage_numbers(usage)
            self.last_usage = (prompt, completion, prompt + completion)
            self.session_prompt_tokens += prompt
            self.session_completion_tokens += completion

        if error:
            self._write_panel(
                "💀 Error",
                Text(error, style=f"bold {BLOOD}"),
                BLOOD,
            )
        elif content and self._assistant_messages_seen == 0:
            body = _clip(_content_to_text(content))
            if body:
                self._write_panel("👻 Agent", body, GHOST)

        self._update_usage()
        self._update_todos()
        self._set_busy(False)

    def _start_compact(self) -> None:
        if len(self.messages) <= 2:
            self._write_system(
                "🧹 Compaction",
                "There is not enough conversation to compact yet.",
                TOXIC,
            )
            return

        self._set_busy(True, "Compacting the grimoire")
        self._compact_turn()

    @work(exclusive=True, thread=True)
    def _compact_turn(self) -> None:
        cost_before = _conversation_cost()
        error: str | None = None
        buffer = io.StringIO()

        try:
            with redirect_stdout(buffer):
                _invoke(compact_history, self.messages)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        cost_delta = max(0.0, _conversation_cost() - cost_before)
        self._post_to_ui(self._compact_finished, cost_delta, error)

    def _compact_finished(
        self,
        cost_delta: float,
        error: str | None,
    ) -> None:
        self.last_turn_cost = float(cost_delta or 0.0)

        if error:
            self._write_panel(
                "💀 Compaction error",
                Text(error, style=f"bold {BLOOD}"),
                BLOOD,
            )

        self._update_usage()
        self._update_todos()
        self._set_busy(False)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_new_conversation(self) -> None:
        if self._busy:
            self._write_system(
                "🕸️ Busy",
                "Cannot start a new conversation while the agent is working.",
                BLOOD,
            )
            return
        self._new_conversation()

    def action_focus_prompt(self) -> None:
        if not self._busy:
            self.query_one("#prompt", Input).focus()

    def action_quit(self) -> None:
        self.exit()


def main() -> None:
    HalloweenAgentApp().run()


if __name__ == "__main__":
    main()
