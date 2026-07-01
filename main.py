
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
version = '1.0'
dark_gray = '\033[38;2;43;43;43m'
terracotta = '\033[38;2;211;124;91m'
beige = '\033[38;2;240;222;201m'
cream = '\033[38;2;245;238;225m'
light_gray = '\033[38;2;168;162;160m'
reset = '\033[0m'
bold = '\033[1m'
hide_cursor = '\033[?25l'
show_cursor = '\033[?25h'


def main():
    try:
        os.system('clear')
        print(hide_cursor, end="")
        print(f"{terracotta}┌─────────────────────────────────────────────────────────────────────────┐{reset}")
        print(f"{bold}{beige}   Добро пожаловать в Vinux CLI!{reset}")
        print(f"{terracotta}└─────────────────────────────────────────────────────────────────────────┘{reset}")
        print(f'''{terracotta}{bold}
██╗   ██╗██╗███╗   ██╗██╗   ██╗██╗  ██╗
██║   ██║██║████╗  ██║██║   ██║╚██╗██╔╝
██║   ██║██║██╔██╗ ██║██║   ██║ ╚███╔╝ 
╚██╗ ██╔╝██║██║╚██╗██║██║   ██║ ██╔██╗ 
 ╚████╔╝ ██║██║ ╚████║╚██████╔╝██╔╝ ██╗
  ╚═══╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝{reset}''')
        print()
        print(show_cursor, end='')
        print(
            f" {dark_gray}│{reset} {cream}Нажмите {bold}{beige}[Enter]{reset}{cream}, чтобы продолжить обучение в системе {terracotta}❯{reset}      {dark_gray}│{reset}")
        input("\033[A\r\033[58C")
        print()

        from uti.main_menu import main_menu
        main_menu(device, version)

    except KeyboardInterrupt:
        print(f"\n\n{terracotta}[Система]{reset} Выход из программы.")
    finally:
        print(show_cursor, end="", flush=True)


if __name__ == '__main__':
    main()