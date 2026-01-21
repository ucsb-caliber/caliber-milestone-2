# Caliber Milestone 2 – PDF Question Parsing & Embeddings

This repository contains a simple Python pipeline to:

1. Parse a PDF exam into structured **question text** using layout detection + OCR
2. Save question metadata and optional image crops to a JSON database
3. Embed all extracted questions using a SentenceTransformer model

---

## Setup (macOS & Windows)

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

## System Dependencies (Required)

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

#### 1. Install Poppler

* Download Poppler for Windows:
  [https://github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases)
* Extract it (e.g. to `C:\poppler`)
* Add `C:\poppler\Library\bin` to your **PATH**

Verify:

```powershell
pdfinfo -v
```

#### 2. Install Tesseract

* Download installer:
  [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
* During install, **check “Add to PATH”**

Verify:

```powershell
tesseract --version
```

---

## Running the Pipeline


From the repo root:

```bash
python server/layoutparser.py
```

You’ll be prompted:

```
Enter exam id:
```

Example:

```
practicefinal3
```


---

## Configuration

### Change input PDF

In `server/test_sample.py`:

```python
PDF_PATH = "exam_tests/practicefinal3.pdf"
```

### Change page range

```python
START_PAGE = 1
END_PAGE = 10
```

### Disable crop display (headless / Windows-friendly)

```python
SHOW_CROPS = False
```
---

