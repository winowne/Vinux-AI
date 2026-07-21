import sys
from textual.app import App
from textual.widgets import Static, OptionList
from textual.containers import Horizontal, Vertical
from textual.widgets.option_list import Option
from textual import work

from config import Config
from src.anime import Rain
from screens.start import StartChatScreen
from screens.history import HistoryScreen

import src.generator as gen


class Poloska(Static):
    def render(self):
        width = self.app.size.width
        return f"┌{'─' * (width - 2)}┐"

class Poloska2(Static):
    def render(self):
        width = self.app.size.width
        return f"├{'─' * (width - 2)}┤"

class LoadProgress(Static):
    def render(self):
        p = gen.load_progress
        if p >= 100:
            return ""
        filled = int(11 * p / 100)
        empty = 11 - filled
        return f"[bold #0178d4]{'█' * filled}[/][bold white]{'█' * empty}[/]"

class MainMenu(App):
    CSS_PATH = 'styles/main.css'

    def compose(self):
        
        with Horizontal(id="top_content"):
            yield Static(Config.logo, id="logo")
            with Vertical(id="sys_info_block"): 

                with Horizontal(classes="info_row"):
                    yield Static(Config.show_os_label, classes="info_text")
                    yield Static(Config.get_os_name(), classes="info_value")
                
                with Horizontal(classes="info_row"):
                    yield Static(Config.show_ver_label, classes="info_text")
                    yield Static(Config.get_version(), classes="info_value")

                with Horizontal(classes="info_row"):
                    yield Static(Config.show_device_label, classes="info_text")
                    yield Static(Config.get_device(), classes="info_value")

                yield LoadProgress(id="load_progress")
                yield Rain(id="matrix_rain")


        yield OptionList(
            Option("История сессий", id="opt_history"),
            Option("Новая сессия", id="opt_new"),
            Option("Выйти", id="opt_exit"),
            id="menu_options")

    def on_mount(self) -> None:
        self.title = Config.app_title
        self.query_one("#menu_options").focus()
        self.load_model_worker()
        self.set_interval(0.2, self.update_progress)

    def update_progress(self) -> None:
        pw = self.query_one("#load_progress", LoadProgress)
        if gen.load_progress >= 100:
            pw.display = False
            return
        pw.refresh()

    @work(exclusive=True, thread=True)
    def load_model_worker(self) -> None:
        gen.load_model_background()
        self.call_from_thread(self.on_model_loaded)

    def on_model_loaded(self) -> None:
        pass

    def on_ready(self) -> None:
        sys.stdout.write(f"\033]0;{Config.app_title}\a")
        sys.stdout.flush()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id == "opt_new":
            self.push_screen(StartChatScreen())
        elif event.option_id == "opt_history":
            self.push_screen(HistoryScreen())
        elif event.option_id == "opt_exit":
            self.exit()

if __name__ == '__main__':
    MainMenu().run()
