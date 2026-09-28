# sps-genai

A FastAPI project with:

- **Bigram text generation** (`/generate`) — from Module 2/3 class activity
- **Word embeddings** (`/embedding`) — added for this assignment, using spaCy's
  pretrained `en_core_web_md` static word vectors

## Setup

```bash
uv sync
uv run python -m spacy download en_core_web_md
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

## Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```
