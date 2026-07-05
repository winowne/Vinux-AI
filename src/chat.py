from pathlib import Path
from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import TextArea, Static
from textual.containers import ScrollableContainer, Container
from textual import work, on
from textual.message import Message
from src.generator import generate_response
from rich.text import Text
import torch

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
    CSS = '''
    #chat-history {
        width: 100%;
        height: 1fr;
        padding: 1 2;
        background: #1c1c1f;
    }

    .bubble-wrapper {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    .user-wrapper {
        align: right middle;
    }

    .ai-wrapper {
        align: left middle;
    }

    .message-bubble {
        border: round #d37c5b;
        color: #f0dec9;
        padding: 0 1;
        width: auto;
        max-width: 70%;
        background: #1c1c1f;
    }

    #chat_input {
        background: #1c1c1f;
        border: round #d37c5b;
        width: 100%;
        height: 3;
    }
    
    #system-info {
        width: 100%;
        height: auto;
        background: #1c1c1f;
        color: #7b7b7b;
        text-align: center;
        padding-bottom: 1;
    }
    
    #suggestions-box {
    background: #2b2b2b;
    border: solid #d37c5b;
    color: #f0dec9;
    width: 60;
    height: auto;
    max-height: 5;
    display: none;
    margin-top: -4;
    align: center middle;
}
    '''

    def __init__(self, initial_message: str = None) -> None:
        super().__init__()
        self.initial_message = initial_message

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="chat-history")
        yield Static("", id="suggestions-box")
        yield ChatInput(id="chat_input")
        yield Static("", id="system-info")

    def on_mount(self) -> None:
        self.query_one("#chat_input").focus()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        module = "Chat"
        version = 'alpha 1.0'

        info_text = Text.from_markup(
            f"[#5f5f5f]│[/] [b][#f0dec9]Устройство -[/] [b][#d37c5b]{device}[/] [#5f5f5f]│[/] "
            f"[b][#f0dec9]Модуль -[/] [b][#d37c5b]{module}[/] [#5f5f5f]│[/] "
            f"[b][#f0dec9]Версия -[/] [b][#d37c5b]{version}[/]"
        )
        self.query_one("#system-info", Static).update(info_text)

        if self.initial_message:
            self.add_bubble(self.initial_message, "user")
            self.ai_inference_task(self.initial_message)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "chat_input":
            return

        current_text = event.text_area.text
        box = self.query_one("#suggestions-box", Static)

        if current_text.startswith("/"):
            commands = ["/exit", "/clear"]
            matches = [cmd for cmd in commands if cmd.startswith(current_text)]

            if matches:
                box.styles.display = "block"
                box.update("\n".join(matches))
            else:
                box.styles.display = "none"
        else:
            box.styles.display = "none"

    def add_bubble(self, text: str, role: str) -> None:
        history = self.query_one("#chat-history", ScrollableContainer)
        wrapper_class = "user-wrapper" if role == "user" else "ai-wrapper"
        wrapper = Container(classes=f"bubble-wrapper {wrapper_class}")
        bubble = Static(text, classes="message-bubble")
        history.mount(wrapper)
        wrapper.mount(bubble)

        bubble.scroll_visible()
        history.refresh(layout=True)

    @on(ChatInput.Submitted)
    def handle_user_message(self, event: ChatInput.Submitted) -> None:
        text = event.text.strip()
        if text.startswith("/"):
            if text == "/exit":
                self.app.exit()
            elif text == "/clear":
                self.query_one("#chat-history", ScrollableContainer).remove_children()
            return

        self.add_bubble(text, "user")
        self.ai_inference_task(text)

    @work(thread=True)
    def ai_inference_task(self, user_text: str) -> None:
        ai_response = generate_response(user_text)
        self.app.call_from_thread(self.add_bubble, ai_response, "ai")