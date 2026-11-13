from typing import Optional, Union

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding import EmbeddingModel
from app.image_classifier import ImageClassifier
from app.image_generators import DiffusionGenerator, EBMGenerator

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
image_classifier = ImageClassifier()
diffusion_generator = DiffusionGenerator()
ebm_generator = EBMGenerator()

PNG_RESPONSE = {200: {"content": {"image/png": {}}, "description": "A PNG grid of generated images."}}


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


@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...), top_k: int = 3):
    """Classify an uploaded image into one of the 10 CIFAR10 classes.

    Example: curl -X POST -F "file=@cat.png" http://127.0.0.1:8000/classify_image
    """
    if not image_classifier.available:
        raise HTTPException(
            status_code=503,
            detail="Model weights not found. Run `uv run python train_cnn.py` first.",
        )
    try:
        return image_classifier.predict(await file.read(), top_k=top_k)
    except Exception:
        raise HTTPException(
            status_code=400, detail="Could not read the uploaded file as an image."
        )


@app.post("/generate_diffusion", response_class=Response, responses=PNG_RESPONSE)
def generate_diffusion(
    num_images: int = Query(4, ge=1, le=16),
    seed: Optional[int] = None,
):
    """Generate CIFAR-10-style images with the diffusion model (returns a PNG grid).

    Example: curl -X POST "http://127.0.0.1:8000/generate_diffusion?num_images=8" -o out.png
    """
    if not diffusion_generator.available:
        raise HTTPException(
            status_code=503,
            detail="Model weights not found. Run `uv run python train_diffusion.py` first.",
        )
    png = diffusion_generator.generate(num_images=num_images, seed=seed)
    return Response(content=png, media_type="image/png")


@app.post("/generate_ebm", response_class=Response, responses=PNG_RESPONSE)
def generate_ebm(
    num_images: int = Query(4, ge=1, le=16),
    steps: int = Query(256, ge=1, le=1000),
    seed: Optional[int] = None,
):
    """Generate CIFAR-10-style images with the energy-based model (returns a PNG grid).

    Starts from random noise and runs `steps` Langevin updates to lower the energy.

    Example: curl -X POST "http://127.0.0.1:8000/generate_ebm?num_images=8" -o out.png
    """
    if not ebm_generator.available:
        raise HTTPException(
            status_code=503,
            detail="Model weights not found. Run `uv run python train_ebm.py` first.",
        )
    png = ebm_generator.generate(num_images=num_images, steps=steps, seed=seed)
    return Response(content=png, media_type="image/png")


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
