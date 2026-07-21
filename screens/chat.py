import json
import re
from pathlib import Path
from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import TextArea, Static
from textual.containers import ScrollableContainer, Container
from textual import work, on
from textual.message import Message
import src.generator as gen
from rich.text import Text
import torch

SESSIONS_DIR = Path("session")

class ChatInput(TextArea):
    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def on_key(self, event) -> None:
        if event.key == "enter":
            event.prevent_default()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.text = ""
        elif event.key == "shift+enter":
            event.prevent_default()
            self.insert_text("\n")


class ChatScreen(Screen):
    CSS_PATH = "../styles/chat.css"

    def __init__(self, initial_message: str = None, session_path: str = None) -> None:
        super().__init__()
        self.initial_message = initial_message
        self.session_path = session_path
        self.session_name = None
        self.messages = []
        self._loading_bubble = None
        self._typewriter_bubble = None
        self._typewriter_text = ""
        self._typewriter_done = True
        self._typewriter_finalized = True

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="chat-history")
        yield Static("", id="suggestions-box")
        yield ChatInput(id="chat_input")
        yield Static("", id="system-info")

    def on_mount(self) -> None:
        self.query_one("#chat_input").focus()
        self.update_token_display()

        if self.session_path:
            self.load_session()
        elif self.initial_message:
            self.add_bubble(self.initial_message, "user")
            self.messages.append({"role": "user", "text": self.initial_message})
            self.show_ai_loading()
            self.ai_inference_task(self.initial_message)

    def load_session(self) -> None:
        path = Path(self.session_path)
        if not path.exists():
            return
        data = json.loads(path.read_text())
        self.session_name = data.get("name")
        self.messages = data.get("messages", [])
        for msg in self.messages:
            self.add_bubble(msg["text"], msg["role"])
        gen.total_input_tokens = data.get("token_input", 0)
        gen.total_output_tokens = data.get("token_output", 0)
        self.update_token_display()

    def save_session(self) -> None:
        SESSIONS_DIR.mkdir(exist_ok=True)
        data = {
            "name": self.session_name,
            "messages": self.messages,
            "token_input": gen.total_input_tokens,
            "token_output": gen.total_output_tokens,
        }
        Path(self.session_path).write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "chat_input":
            return

        current_text = event.text_area.text
        box = self.query_one("#suggestions-box", Static)

        if current_text.startswith("/"):
            commands = ["/exit", "/clear", "/back"]
            matches = [cmd for cmd in commands if cmd.startswith(current_text)]

            if matches:
                box.styles.display = "block"
                box.update("\n".join(matches))
            else:
                box.styles.display = "none"
        else:
            box.styles.display = "none"

    def add_bubble(self, text: str, role: str) -> Static:
        history = self.query_one("#chat-history", ScrollableContainer)
        wrapper_class = "user-wrapper" if role == "user" else "ai-wrapper"
        wrapper = Container(classes=f"bubble-wrapper {wrapper_class}")
        bubble = Static(text, classes="message-bubble")
        history.mount(wrapper)
        wrapper.mount(bubble)

        bubble.scroll_visible()
        history.refresh(layout=True)
        return bubble

    def show_ai_loading(self) -> None:
        bubble = self.add_bubble("ИИ думает...", "ai")
        self._loading_bubble = bubble

    def hide_ai_loading(self) -> None:
        if self._loading_bubble:
            try:
                self._loading_bubble.parent.remove()
            except Exception:
                pass
            self._loading_bubble = None

    def update_token_display(self) -> None:
        total = gen.total_input_tokens + gen.total_output_tokens
        sys_info = self.query_one("#system-info", Static)
        sys_info.update(
            f"Входящие токены: {gen.total_input_tokens} | "
            f"Исходящие токены: {gen.total_output_tokens} | "
            f"Всего: {total}"
        )

    def _complete_typewriter(self) -> None:
        if self._typewriter_done:
            return
        self._typewriter_done = True
        self._typewriter_bubble.update(self._typewriter_text)
        self._typewriter_bubble.scroll_visible()
        self._finalize_ai_response()

    @on(ChatInput.Submitted)
    def handle_user_message(self, event: ChatInput.Submitted) -> None:
        text = event.text.strip()

        self._complete_typewriter()

        if text.startswith("/"):
            if text == "/exit":
                self.app.exit()
            elif text == "/clear":
                self.query_one("#chat-history", ScrollableContainer).remove_children()
                self.messages = []
                self._loading_bubble = None
                self._typewriter_done = True
                self._typewriter_finalized = True
                gen.reset_token_counts()
                self.update_token_display()
            elif text == "/back":
                self._complete_typewriter()
                while len(self.app.screen_stack) > 1:
                    self.app.pop_screen()
            return

        self.add_bubble(text, "user")
        self.messages.append({"role": "user", "text": text})
        self.show_ai_loading()
        self.ai_inference_task(text)

    @work(thread=True)
    def ai_inference_task(self, user_text: str) -> None:
        ai_response = gen.generate_response(user_text)
        self.app.call_from_thread(self.on_ai_response, ai_response)

    def on_ai_response(self, ai_response: str) -> None:
        self.hide_ai_loading()
        bubble = self.add_bubble("", "ai")
        self._typewriter_bubble = bubble
        self._typewriter_text = ai_response
        self._typewriter_done = False
        self._typewriter_finalized = False
        self._typewriter_index = 0
        self.set_timer(0.02, self._typewriter_tick)

    def _typewriter_tick(self) -> None:
        if self._typewriter_done:
            return
        self._typewriter_index += 1
        if self._typewriter_index <= len(self._typewriter_text):
            self._typewriter_bubble.update(self._typewriter_text[:self._typewriter_index])
            self._typewriter_bubble.scroll_visible()
            self.set_timer(0.02, self._typewriter_tick)
        else:
            self._typewriter_done = True
            self._finalize_ai_response()

    def _finalize_ai_response(self) -> None:
        if self._typewriter_finalized:
            return
        self._typewriter_finalized = True
        self.messages.append({"role": "ai", "text": self._typewriter_text})

        if not self.session_path:
            safe_name = re.sub(r'[^\w\-_\. ]', '_', self._typewriter_text)[:60]
            self.session_name = self._typewriter_text
            self.session_path = str(SESSIONS_DIR / f"{safe_name}.json")

        self.save_session()
        self.update_token_display()
