import spacy


class EmbeddingModel:
    """Wraps a spaCy pipeline to provide word embedding lookups.

    Uses a medium/large spaCy model (e.g. en_core_web_md) so that
    `.vector` returns a pretrained static word embedding rather than
    the zero-vector fallback of the small model.
    """

    def __init__(self, model_name: str = "en_core_web_md"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. "
                f"Run: python -m spacy download {model_name}"
            ) from exc

    def get_embedding(self, word: str) -> list[float]:
        token = self.nlp(word)[0]
        return token.vector.tolist()

    def has_vector(self, word: str) -> bool:
        token = self.nlp(word)[0]
        return bool(token.has_vector) and not token.is_oov

    def most_similar(self, word: str, topn: int = 5) -> list[dict]:
        """Return the topn words in the vocabulary closest to `word`
        by cosine similarity, using spaCy's vector table.
        """
        query = self.nlp.vocab[word]
        if not query.has_vector:
            return []

        by_similarity = []
        for lexeme in self.nlp.vocab:
            if not lexeme.has_vector or not lexeme.is_alpha or not lexeme.is_lower:
                continue
            if len(lexeme.text) < 3 or lexeme.text.lower() == word.lower():
                continue
            similarity = query.similarity(lexeme)
            by_similarity.append((lexeme.text, similarity))

        by_similarity.sort(key=lambda pair: pair[1], reverse=True)
        return [
            {"word": w, "similarity": float(s)} for w, s in by_similarity[:topn]
        ]
