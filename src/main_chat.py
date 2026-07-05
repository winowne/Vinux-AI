from importlib.metadata import version
from pathlib import Path
from src.chat import ChatScreen
from textual.app import App, ComposeResult
from textual.widgets import TextArea, Static
from textual.events import Key
import torch
from rich.text import Text

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
version = 'alpha 1.0'


class StartInput(TextArea):
    def on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            text = self.text.strip()
            if not text: return

            if text == "/exit":
                self.app.exit()
                return

            self.app.push_screen(ChatScreen(initial_message=text))
            self.text = ""
        elif event.key == "shift+enter":
            event.prevent_default()
            self.insert_text("\n")

class VinuxApp(App):
    CSS = '''
        Screen {
            background: #1c1c1f;
            align: center middle;
        }
        #suggestions-box {
            background: #2b2b2b;
            border: solid #d37c5b;
            color: #f0dec9;
            width: 60;
            height: auto;
            max-height: 5;
            margin-bottom: 0;
            display: none;
        }
        #user_input {
            background: #1c1c1f;
            border: round #d37c5b;
            width: 60;
            height: auto;
            max-height: 5;
            margin-top: 0;    
            padding: 0 1;
        }
        #system-info {
            width: 60;
            height: auto;
            color: #7b7b7b;
            margin-top: 0;   
            text-align: center;
        }
        '''

    def compose(self) -> ComposeResult:
        yield Static("", id="suggestions-box")
        yield StartInput(placeholder='Enter your text here', id="user_input")
        yield Static("", id="system-info")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        current_text = event.text_area.text
        box = self.query_one("#suggestions-box", Static)

        if current_text.startswith("/"):
            commands = ["/exit"]
            matches = [cmd for cmd in commands if cmd.startswith(current_text)]

            if matches:
                box.styles.display = "block"
                box.update("\n".join(matches))
            else:
                box.styles.display = "none"
        else:
            box.styles.display = "none"

    def on_mount(self) -> None:
        self.query_one("#user_input").focus()

        device = device_name
        module = "Chat"

        info_text = Text.from_markup(
            "[#5f5f5f]│[/] [b][#f0dec9]Устройство -[/] [b][#d37c5b]" + device + "[/] [#5f5f5f]│[/] "
            "[b][#f0dec9]Модуль -[/] [b][#d37c5b]" + module + "[/] [#5f5f5f]│[/] "
            "[b][#f0dec9]Версия -[/] [b][#d37c5b]" + version + "[/]"
        )
        self.query_one("#system-info", Static).update(info_text)