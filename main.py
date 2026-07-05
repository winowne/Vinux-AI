import os
from pathlib import Path
from src.main_chat import VinuxApp
import sys

def set_terminal_title(title):
    sys.stdout.write(f"\033]0;{title}\007")
    sys.stdout.flush()

set_terminal_title("vinux")


dark_gray = '\033[38;2;43;43;43m'
terracotta = '\033[38;2;211;124;91m'
beige = '\033[38;2;240;222;201m'
cream = '\033[38;2;245;238;225m'
light_gray = '\033[38;2;168;162;160m'
reset = '\033[0m'
bold = '\033[1m'

logo = f'''{terracotta}{bold}
  ██   ██ ██ ███  ██ ██ ██ ██  ██
   ██ ██  ██ ██ █ ██ ██ ██   ██
    ███   ██ ██  ███  ███  ██  ██{reset}
'''

def main():
    try:
        os.system('clear')
        print(f'{terracotta}┌─────────────────────────────────────────────────────────────────────────┐{reset}')
        print(f'{dark_gray}│ {cream}Добро пожаловать в {bold}Vinux{reset}!{cream}{reset}')
        print(f'{terracotta}└─────────────────────────────────────────────────────────────────────────┘{reset}')
        print(logo)
        enter = input(f" {dark_gray}│{reset} {cream}Нажмите {bold}[Enter]{reset}{cream}, чтобы продолжить >>>{reset}  ")
        os.system('clear')
        VinuxApp().run()

    except KeyboardInterrupt:
        os.system('clear')

if "__main__" == __name__:
    main()