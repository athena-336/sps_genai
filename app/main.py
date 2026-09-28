from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding import EmbeddingModel

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel()


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str
    top_n: int = 5
    include_similar: bool = False


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.get("/embedding")
def get_embedding(word: str, include_similar: bool = False, top_n: int = 5):
    """Return the word embedding vector for a single query word.

    Example: GET /embedding?word=king
    Example: GET /embedding?word=king&include_similar=true&top_n=5
    """
    if not embedding_model.has_vector(word):
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for '{word}' (out of vocabulary).",
        )

    response = {
        "word": word,
        "embedding": embedding_model.get_embedding(word),
        "dimensions": len(embedding_model.get_embedding(word)),
    }

    if include_similar:
        response["most_similar"] = embedding_model.most_similar(word, topn=top_n)

    return response


@app.post("/embedding")
def post_embedding(request: EmbeddingRequest):
    """Same functionality as GET /embedding, but accepts a JSON body."""
    if not embedding_model.has_vector(request.word):
        raise HTTPException(
            status_code=404,
            detail=f"No embedding found for '{request.word}' (out of vocabulary).",
        )

    response = {
        "word": request.word,
        "embedding": embedding_model.get_embedding(request.word),
        "dimensions": len(embedding_model.get_embedding(request.word)),
    }

    if request.include_similar:
        response["most_similar"] = embedding_model.most_similar(
            request.word, topn=request.top_n
        )

    return response
