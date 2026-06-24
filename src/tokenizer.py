import json

class Vinux:
    def __init__(self):
        try:
            with open('data/vocab.json', 'r', encoding='utf-8') as file:
                self.vocab = json.load(file)
        except FileNotFoundError:
            print('не найден vocab.json')
            with open('data/data.txt','r', encoding='utf-8') as f:
                text = f.read()

                text = text.lower()

                symbols = """.,!?;:()[]{}"'«»—_…-–+=*&^%$#@`~|/\\<>"""
                for s in symbols:
                    text = text.replace(s, f" {s} ")

                words = text.split()
                unique_words = set(words)
                vocab = {}
                id_word = 0

            for word in unique_words:
                vocab[word] = id_word
                id_word += 1

            with open('data/vocab.json', 'w', encoding='utf-8') as file:
                json.dump(vocab, file, ensure_ascii=False, indent=4)

            with open('data/vocab.json', 'r', encoding='utf-8') as file:
                self.vocab = json.load(file)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        text = text.lower()

        symbols = """.,!?;:()[]{}"'«»-…"""
        for s in symbols:
            text = text.replace(s, f" {s} ")

        encoded_ids = []
        words = text.split()
        for word in words:
            if word in self.vocab:
                encoded_ids.append(self.vocab[word])
            else:
                continue

        return encoded_ids

    def decode(self, ids: list[int]) -> str:
        words = []
        for token_id in ids:
            words.append(self.inv_vocab[token_id])
        return " ".join(words)

if __name__ == "__main__":
    Vinux()