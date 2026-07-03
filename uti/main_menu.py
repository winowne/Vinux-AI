import os
import time
import string
import sys
import itertools
import threading
from src.generator import generate_response

dark_gray = '\033[38;2;43;43;43m'
terracotta = '\033[38;2;211;124;91m'
beige = '\033[38;2;240;222;201m'
cream = '\033[38;2;245;238;225m'
light_gray = '\033[38;2;168;162;160m'
reset = '\033[0m'
bold = '\033[1m'
hide_cursor = '\033[?25l'
show_cursor = '\033[?25h'

model = 'tigr'
stop_animation = False


def animate_thinking():
    global stop_animation
    for char in itertools.cycle(['|', '/', '-', '\\']):
        if stop_animation:
            break
        sys.stdout.write(f'\r{char}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r' + ' ' * 30 + '\r')


def main_menu(device, version):
    global stop_animation
    os.system('clear')
    print(f"{terracotta}┌─────────────────────────────────────────────────────────────────────────┐{reset}")
    print(
        f'{dark_gray}│{reset} {bold}{beige}Модель - {model}{reset}  {dark_gray}│{reset}  {bold}{beige}Устройство - {device}  {dark_gray}│{reset}')
    print(f"{terracotta}└─────────────────────────────────────────────────────────────────────────┘{reset}")
    print()

    while True:
        print(f'{light_gray}┌─────────────────────────────────────────────────────────────────────────┐{reset}')
        print(
            f'{light_gray}│{reset}  {bold}{terracotta}>>>{reset}                                                                    {light_gray}│{reset}')
        print(f"{light_gray}└─────────────────────────────────────────────────────────────────────────┘{reset}")

        print(show_cursor, end="")
        user_text = input("\033[2A\r\033[9C").strip()

        if user_text.lower() in ['exit', 'выход']:
            print(show_cursor)
            break
        if not user_text:
            print("\033[2B")
            continue

        print(hide_cursor, end="")
        print("\033[2B\r", end="")
        print()

        stop_animation = False
        t = threading.Thread(target=animate_thinking)
        t.start()

        bot_output = generate_response(user_text)

        stop_animation = True
        t.join()

        sys.stdout.write('\r\033[K')
        print(f' {bold}{beige}Vinux >>>{reset} {cream}{bot_output}{reset}')
        print()