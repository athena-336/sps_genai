# sps-genai

A FastAPI project with:

- **Bigram text generation** (`/generate`) — from Module 2/3 class activity
- **Word embeddings** (`/embedding`) — added for this assignment, using spaCy's
  pretrained `en_core_web_md` static word vectors

- **CNN image classifier** (`/classify_image`) — added for Assignment 2, a
  PyTorch CNN trained on CIFAR10 (images resized to 64x64)

## Setup

```bash
uv sync
uv run python -m spacy download en_core_web_md
uv run python train_cnn.py        # trains the CNN, saves app/cnn_cifar10.pt
uv run fastapi dev app/main.py
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## Endpoints

### `GET /`
Health check.

### `POST /generate`
Generate text from the bigram model.

```json
{
  "start_word": "this",
  "length": 6
}
```

### `GET /embedding?word=king&include_similar=false&top_n=5`
### `POST /embedding`

Returns the 300-dimensional spaCy word embedding for a single query word.

```json
{
  "word": "king",
  "include_similar": true,
  "top_n": 5
}
```

Response:

```json
{
  "word": "king",
  "embedding": [ -0.606, -0.512, ... ],
  "dimensions": 300,
  "most_similar": [
    {"word": "queen", "similarity": 0.72},
    ...
  ]
}
```

Returns `404` if the query word is out of vocabulary (no pretrained vector).

**Note:** `en_core_web_md` maps ~685k words down to ~20k unique vectors via
hashing, so `most_similar` results can be noisier than with `en_core_web_lg`,
which has a full unique vector table. Swap the model name in
`app/embedding.py` for `en_core_web_lg` for higher-quality nearest-neighbor
results, at the cost of a larger download.

### `POST /classify_image`

Upload an image (multipart form field `file`) and get the predicted CIFAR10
class. Optional query parameter `top_k` (default 3).

```bash
curl -X POST -F "file=@cat.png" http://127.0.0.1:8000/classify_image
```

Response:

```json
{
  "class": "cat",
  "confidence": 0.71,
  "top_predictions": [
    {"class": "cat", "confidence": 0.71},
    {"class": "dog", "confidence": 0.18},
    {"class": "deer", "confidence": 0.04}
  ]
}
```

Returns `400` if the file is not a readable image, and `503` if the trained
weights (`app/cnn_cifar10.pt`) have not been created yet.

**CNN architecture** (`app/cnn_model.py`): input 3x64x64 → Conv(16, 3x3, s1, p1)
→ ReLU → MaxPool(2, s2) → Conv(32, 3x3, s1, p1) → ReLU → MaxPool(2, s2) →
Flatten (8192) → Linear(100) → ReLU → Linear(10).

## Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```
