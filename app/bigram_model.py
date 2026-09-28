import random
import re
from collections import defaultdict


class BigramModel:
    """A simple bigram language model built from a small text corpus.

    Learns P(next_word | current_word) counts from the corpus and can
    generate new text by sampling from those bigram probabilities.
    """

    def __init__(self, corpus: list[str]):
        self.bigrams: dict[str, list[str]] = defaultdict(list)
        self._train(corpus)

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        return re.findall(r"[a-z']+", text)

    def _train(self, corpus: list[str]) -> None:
        for sentence in corpus:
            tokens = self._tokenize(sentence)
            for current_word, next_word in zip(tokens, tokens[1:]):
                self.bigrams[current_word].append(next_word)

    def generate_text(self, start_word: str, length: int) -> str:
        current_word = start_word.lower()
        generated = [current_word]

        for _ in range(length - 1):
            next_words = self.bigrams.get(current_word)
            if not next_words:
                break
            current_word = random.choice(next_words)
            generated.append(current_word)

        return " ".join(generated)
