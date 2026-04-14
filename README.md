# Caliber Milestone 2

This repository contains a Python pipeline to:

1. Parse a PDF exam into structured **question text** using layout detection + a local VLM (via Ollama)
2. Save question metadata and image crops to a JSON database
3. (Optional) Embed extracted questions downstream

Layout detection identifies question boundaries. Crops of each question are sent to a local vision language model (VLM) which returns the content as structured Markdown — replacing traditional OCR.

---

## Prerequisites

### 1. Ollama (local VLM)

Install Ollama: https://ollama.com

Pull the vision model:

```bash
ollama pull qwen2.5vl:7b
```

Start the server (runs in background):

```bash
ollama serve
```

To use a different model:

```bash
export OLLAMA_MODEL=llava
```

To point at a remote Ollama instance:

```bash
export OLLAMA_URL=http://your-host:11434
```

---

### 2. Poppler (PDF rendering)

**macOS**

```bash
brew install poppler
```

**Windows**

- Download: https://github.com/oschwartz10612/poppler-windows/releases
- Extract and add `bin/` folder to PATH

---

## Setup

### 1. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

From the repo root:

```bash
MPLBACKEND=Agg python server/layout_ingest.py
```

You'll be prompted:

```text
Enter exam id:
```

Outputs (JSON + crops) are written to `layout_debug/`.

---

## Configuration

All config lives at the top of `server/layout_ingest.py`:

| Variable | Default | Description |
|---|---|---|
| `PDF_PATH` | `exam_tests/practicefinal3.pdf` | Input PDF |
| `START_PAGE` | `1` | First page to process |
| `END_PAGE` | `10` | Last page to process (0 = all) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL (env-overridable) |
| `OLLAMA_MODEL` | `qwen2.5vl:7b` | Vision model to use (env-overridable) |
| `SAVE_CROPS` | `True` | Save question crop images |
| `SHOW_CROPS` | `True` | Display crops (set False for headless) |

---

## How It Works

1. **Layout detection** — EfficientDet/Detectron2 (PubLayNet) detects block bounding boxes and types (`Title`, `Text`, `List`, `Figure`, `Table`)
2. **Question grouping** — `Title` blocks mark question boundaries; subsequent blocks accumulate under the current question
3. **Cropping** — The merged bounding box of each question is cropped from the page image
4. **VLM extraction** — Each crop is sent to Ollama as a base64 PNG; the model returns the question content as Markdown
5. **Storage** — Questions are written to `layout_debug/questions.json` with IDs, page numbers, crop paths, and Markdown text

---

## Output Format

`layout_debug/questions.json`:

```json
{
  "schema_version": "1.0",
  "ingestions": [
    {
      "ingestion_id": "ing_...",
      "exam_id": "practicefinal3",
      "questions": [
        {
          "question_id": "q_...",
          "start_page": 1,
          "page_nums": [1, 2],
          "text": "## Problem 1\n\nFor each of the following...",
          "image_crops": ["layout_debug/crops/.../q_..._p001.png"],
          "type": null,
          "metadata": {}
        }
      ]
    }
  ]
}
```

---

## Docker

```bash
docker build -t caliber-layout-ingest .

docker run --rm -it \
  -v "$PWD:/app" \
  caliber-layout-ingest
```

> Note: Ollama must be accessible from within the container. Set `OLLAMA_URL` to your host IP if running Ollama on the host machine.

---

## Troubleshooting

**"Could not connect to Ollama"** — Make sure `ollama serve` is running and the model is pulled.

**Detectron2 warning on startup** — Normal. The pipeline falls back to EfficientDet automatically.

**Rebuild Docker with no cache:**

```bash
docker build --no-cache -t caliber-layout-ingest .
```
