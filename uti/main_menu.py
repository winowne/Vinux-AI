import os
import time
import json
import random
from src.gen_for_tiki import generate_response
import string

dark_gray = '\033[38;2;43;43;43m'
terracotta = '\033[38;2;211;124;91m'
beige = '\033[38;2;240;222;201m'
cream = '\033[38;2;245;238;225m'
light_gray = '\033[38;2;168;162;160m'
reset = '\033[0m'
bold = '\033[1m'
dim = '\033[2m'
hide_cursor = '\033[?25l'
show_cursor = '\033[?25h'

model = 'tiki'

try:
    with open('data/datasets.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    intent_answers = {}
    responses_pool = data.get('responses', {})

    for intent in data.get('intents', []):
        label = intent['label']
        answers_list = responses_pool.get(label, None)

        if answers_list:
            for phrase in intent.get('phrases', []):
                intent_answers[phrase.lower().strip()] = answers_list

except Exception:
    intent_answers = {}


def main_menu(device, version):
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
        print(f"{dim}{dark_gray} ⏳ Vinux думает...{reset}", end="\r")

        clean_input = user_text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

        if clean_input in intent_answers and intent_answers[clean_input]:
            time.sleep(0.05)
            bot_output = random.choice(intent_answers[clean_input])
        else:
            bot_output = generate_response(user_text)

        print("\033[K", end="")
        print(f' {bold}{beige}Vinux >>>{reset} {cream}{bot_output}{reset}')
        print()