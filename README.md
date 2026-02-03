````md
# Caliber Milestone 2

This repository contains a Python pipeline to:

1. Parse a PDF exam into structured **question text** using layout detection + OCR  
2. Save question metadata and optional image crops to a JSON database  
3. (Optional) Embed extracted questions downstream  

The pipeline can be run **locally** or **via Docker (recommended for consistency)**.

---

## Recommended: Run with Docker (Cross-platform, reproducible)

Docker avoids OS-specific issues with:
- Tesseract / Poppler installation
- PyTorch + Detectron2 compatibility
- macOS vs Windows vs Linux differences

### Prerequisites
- Docker Desktop installed  
  https://www.docker.com/products/docker-desktop/

---

### Build the Docker image

From the repo root:

```bash
docker build -t caliber-layout-ingest .
````

---

### Run the pipeline in Docker

```bash
docker run --rm -it \
  -v "$PWD:/app" \
  caliber-layout-ingest
```

You’ll be prompted:

```text
Enter exam id:
```

Example:

```text
hw3
```

Outputs (JSON + crops) are written to:

```text
layout_debug/
```

because the repo is mounted into the container.

---

### Optional Docker flags

Disable crop display (recommended in Docker):

```bash
docker run --rm -it \
  -v "$PWD:/app" \
  -e SHOW_CROPS=0 \
  caliber-layout-ingest
```

---

## Local Setup (macOS & Windows)

> ⚠️ Local setup is more fragile due to system dependencies.
> Use Docker unless you explicitly need a local environment.

---

### 1. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## System Dependencies (Local Only)

This project uses **OCR and PDF rendering**, which require system tools.

---

### macOS (Homebrew)

Install Homebrew if needed: [https://brew.sh](https://brew.sh)

```bash
brew install poppler tesseract
```

Verify:

```bash
which pdfinfo
which tesseract
```

---

### Windows

#### Install Poppler

* Download:
  [https://github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases)
* Extract (e.g. `C:\poppler`)
* Add `C:\poppler\Library\bin` to **PATH**

Verify:

```powershell
pdfinfo -v
```

---

#### Install Tesseract

* Download:
  [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
* During install, **check “Add to PATH”**

Verify:

```powershell
tesseract --version
```

---

## Running the Pipeline (Local)

From the repo root:

```bash
python server/layout_ingest.py
```

You’ll be prompted:

```text
Enter exam id:
```

Example:

```text
practicefinal3
```

---

## Configuration

### Change input PDF

In `server/layout_ingest.py`:

```python
PDF_PATH = "exam_tests/practicefinal3.pdf"
```

---

### Change page range

```python
START_PAGE = 1
END_PAGE = 10
```

---

### Disable crop display

```python
SHOW_CROPS = False
```

Or (Docker-friendly):

```bash
export SHOW_CROPS=0
```

---

## Notes

* Docker uses **CPU-only execution** by default.
* Detectron2 is preferred when available; EfficientDet is used as a fallback.
* All outputs are deterministic when run in Docker.

---

## Troubleshooting

If something works locally but not in Docker:

* Rebuild with no cache:

  ```bash
  docker build --no-cache -t caliber-layout-ingest .
  ```

* Ensure you removed any OS-specific shell calls (`ip route`, etc.).

* Ensure there is **no local file named `layoutparser.py`** (this breaks imports).

---

## Recommendation

For team development and grading:

> **Use Docker.**
> It eliminates OS drift and “works on my machine” bugs.

```
```
