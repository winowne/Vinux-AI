import subprocess
from pathlib import Path
from textual.widgets import Static
from textual.events import MouseDown

maskot = '''
   █    
   █   █
   █████
   █ █ █
  ███████
   █████
   █   █
'''

class Mascot(Static):
    def on_mount(self) -> None:
        self._jump_interval = 1.0
        self._jump_timer = self.set_interval(self._jump_interval, self._jump)

    def _jump(self) -> None:
        self.styles.offset = (0, -3)
        self.set_timer(0.15, self._land)

    def _land(self) -> None:
        self.styles.offset = (0, 0)

    def play_squeak(self) -> None:
        sound_path = Path(__file__).parent.parent / "sounds" / "squeak.mp3"
        subprocess.Popen(
            ['ffplay', '-nodisp', '-autoexit', str(sound_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def on_mouse_down(self, event: MouseDown) -> None:
        if event.button == 1:
            self._jump_interval = max(0.5, self._jump_interval / 1.5)
            self._jump_timer.stop()
            self._jump_timer = self.set_interval(self._jump_interval, self._jump)
        elif event.button == 3:
            self.play_squeak()
