from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import TextArea, Static
from textual.events import Key
from screens.chat import ChatScreen
from src.maskot import Mascot, maskot

class StartInput(TextArea):
    def on_key(self, event: Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            text = self.text.strip()
            if not text: 
                return

            if text == "/exit":
                self.app.exit()
                return

            self.app.push_screen(ChatScreen(initial_message=text))
            self.text = ""
        elif event.key == "shift+enter":
            event.prevent_default()
            self.insert_text("\n")

class StartChatScreen(Screen):
    CSS_PATH = "../styles/start.css"

    def compose(self) -> ComposeResult:
        yield Static("", id="suggestions-box")
        yield Mascot(maskot, id="mascot")
        yield StartInput(placeholder='Введите ваш запрос здесь...', id="user_input")
        

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