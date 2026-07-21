import random
from textual.widgets import Static
from config import Config

class Rain(Static):
    def on_mount(self) -> None:
        self.width = getattr(Config, "rain_width", 15)
        self.height = getattr(Config, "rain_height", 1)
        self.lines = [self.generate_line() for _ in range(self.height)]
        self.set_interval(0.2, self.shift_down)

    def generate_line(self) -> str:
        bits = [str(random.randint(0, 1)) for _ in range(self.width)]
        return " ".join(bits)

    def shift_down(self) -> None:
        self.lines.pop()
        self.lines.insert(0, self.generate_line())
        
        self.update("\n".join(self.lines))