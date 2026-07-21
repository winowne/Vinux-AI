import json
from pathlib import Path
from textual import events
from textual.screen import Screen
from textual.app import ComposeResult
from textual.widgets import Static, OptionList
from textual.widgets.option_list import Option
from screens.chat import ChatScreen
from screens.start import StartChatScreen
from textual.containers import Horizontal

SESSIONS_DIR = Path("session")

class HistoryScreen(Screen):
    CSS_PATH = "../styles/history.css"

    def compose(self) -> ComposeResult:
        yield Static("История сессий", id="title")
        yield Static("", id="no-sessions-msg")
        with Horizontal(id="btn-row"):
            yield Static("Да", id="create_yes")
            yield Static("Нет", id="create_no")
        yield OptionList(id="session_list")

    def _add_empty_prompt(self) -> None:
        self.query_one("#no-sessions-msg", Static).update(
            "Нет сохранённых сессий, хотите создать новую?"
        )
        self.query_one("#btn-row", Horizontal).styles.display = "block"
        self.query_one("#session_list", OptionList).styles.display = "none"

    def on_mount(self) -> None:
        SESSIONS_DIR.mkdir(exist_ok=True)
        option_list = self.query_one("#session_list", OptionList)
        files = sorted(SESSIONS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)

        if not files:
            self._add_empty_prompt()
        else:
            self.query_one("#btn-row", Horizontal).styles.display = "none"
            for f in files:
                data = json.loads(f.read_text())
                name = data.get("name", f.stem)
                t_in = data.get("token_input", 0)
                t_out = data.get("token_output", 0)
                label = f"{name}  [{t_in}|{t_out}]" if t_in or t_out else name
                option_list.add_option(Option(label, id=f.name))
            option_list.focus()

    def on_click(self, event: events.Click) -> None:
        if not event.widget:
            return

        if event.widget.id == "create_yes":
            self.app.push_screen(StartChatScreen())
        elif event.widget.id == "create_no":
            self.app.exit()
