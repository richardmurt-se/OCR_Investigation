# OCR Investigation

A system for comparing OCR models against each other using a synthetic invoice dataset. Currently supports Azure Document Intelligence, PaddleOCR, and GLM-OCR, with a unified evaluation pipeline and cross-model comparison reports.

## Features

- Process invoice documents (PDF, PNG, JPEG) through multiple OCR models
- **Azure Document Intelligence**: Run locally via CLI with structured field extraction
- **PaddleOCR**: Lightweight OCR run on Kaggle GPU (CPU mode)
- **GLM-OCR**: Vision-language model run on Kaggle GPU
- Per-model HTML/JSON reports and a cross-model comparison dashboard

## Prerequisites

- Python 3.9 or higher
- [Poetry](https://python-poetry.org/) for dependency management
- [Kaggle CLI](https://github.com/Kaggle/kaggle-api) and account with GPU access (for PaddleOCR and GLM-OCR)

## Setup

### 1. Install dependencies

```bash
pip install poetry
poetry install
```

### 2. Configure Azure credentials

Create a `.env` file in the project root:

```
AZURE_FORM_RECOGNIZER_KEY=your_api_key_here
AZURE_FORM_RECOGNIZER_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
```

### 3. Configure Kaggle CLI (for PaddleOCR and GLM-OCR)

1. Create a Kaggle account at [kaggle.com](https://www.kaggle.com) and verify your phone number (required for GPU access)
2. Go to Settings > API > "Create New Token" to download `kaggle.json`
3. Place `kaggle.json` at:
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`
4. Upload your invoice dataset to Kaggle as a private dataset

See `Kaggle.md` for detailed Kaggle setup instructions.

## Running OCR Models

### Azure Document Intelligence (Local CLI)

Run OCR on all documents locally using the Azure API:

```bash
# Process documents (skips already-processed by default)
poetry run python scripts/run_ocr.py

# Process with a limit
poetry run python scripts/run_ocr.py --limit 10

# Force reprocessing of all documents
poetry run python scripts/run_ocr.py --force

# Test a single document first
poetry run python scripts/test_single.py
poetry run python scripts/test_single.py dataset/001_clean_standard.png --verbose
```

Results are saved to `results/document_intelligence/` with raw API responses in `results/document_intelligence/raw/`.

### PaddleOCR (Kaggle Notebook)

PaddleOCR runs on Kaggle's infrastructure using `scripts/paddle_ocr.ipynb`.

**1. Configure and push the notebook:**

Edit `scripts/kernel-metadata.json` to point to the PaddleOCR notebook:

```json
{
  "id": "{your_kaggle_username}/paddleocr-invoice-ocr",
  "title": "PaddleOCR Invoice OCR",
  "code_file": "paddle_ocr.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["{your_kaggle_username}/synthetic-invoices"]
}
```

```bash
kaggle kernels push -p ./scripts/
```

**2. Monitor and download results:**

```bash
# Check status
kaggle kernels status {your_kaggle_username}/paddleocr-invoice-ocr

# Download results when complete
kaggle kernels output {your_kaggle_username}/paddleocr-invoice-ocr -p ./results/paddle_ocr/
```

### GLM-OCR (Kaggle Notebook)

GLM-OCR runs on Kaggle's infrastructure using `scripts/glm_ocr.ipynb`.

**1. Configure and push the notebook:**

Edit `scripts/kernel-metadata.json` to point to the GLM-OCR notebook:

```json
{
  "id": "{your_kaggle_username}/glm-ocr-invoice-ocr",
  "title": "GLM-OCR Invoice OCR",
  "code_file": "glm_ocr.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["{your_kaggle_username}/synthetic-invoices"]
}
```

```bash
kaggle kernels push -p ./scripts/
```

**2. Monitor and download results:**

```bash
# Check status
kaggle kernels status {your_kaggle_username}/glm-ocr-invoice-ocr

# Download results when complete
kaggle kernels output {your_kaggle_username}/glm-ocr-invoice-ocr -p ./results/glm_ocr/
```

## Generating Reports

### Per-Model Report

After downloading results for any model, generate a comparison report against ground truth:

```bash
# Azure Document Intelligence
poetry run python scripts/generate_report.py --model document_intelligence

# PaddleOCR
poetry run python scripts/generate_report.py --model paddle_ocr

# GLM-OCR
poetry run python scripts/generate_report.py --model glm_ocr
```

This produces a JSON report and an HTML report in the `reports/` directory.

### Cross-Model Comparison

Generate a dashboard comparing all models that have reports:

```bash
poetry run python scripts/generate_cross_model_report.py
```

This auto-discovers all `*_comparison_report.json` files and produces `reports/cross_model_comparison.html` with charts for extraction accuracy, effect types, file formats, languages, and performance.

To compare specific models only:

```bash
poetry run python scripts/generate_cross_model_report.py --models document_intelligence,paddle_ocr
```

## Project Structure

```
OCR_Investigation/
├── src/
│   ├── models/
│   │   ├── base.py                         # Abstract OCR interface
│   │   └── document_intelligence.py        # Azure DI implementation
│   ├── evaluation/
│   │   ├── compare.py                      # Result comparison logic
│   │   └── metrics.py                      # Accuracy calculations
│   ├── reporting/
│   │   └── html_report.py                  # Per-model HTML report generator
│   ├── templates/
│   │   ├── report.html                     # Per-model report template
│   │   └── cross_model_report.html         # Cross-model comparison template
│   └── config.py                           # Settings, pricing, env loader
├── scripts/
│   ├── test_single.py                      # Test with single document
│   ├── run_ocr.py                          # Batch OCR processing
│   ├── generate_report.py                  # Per-model report generation
│   ├── generate_cross_model_report.py      # Cross-model comparison report
│   ├── paddle_ocr.ipynb                    # PaddleOCR Kaggle notebook
│   ├── glm_ocr.ipynb                       # GLM-OCR Kaggle notebook
│   ├── deepseek_ocr2.ipynb                 # DeepSeek-OCR-2 Kaggle notebook
│   └── kernel-metadata.json                # Kaggle CLI push config (gitignored)
├── dataset/                                # Invoice documents (PDF, PNG, JPEG)
├── results/                                # OCR outputs per model
│   ├── document_intelligence/              # Azure DI results + raw/
│   ├── paddle_ocr/                         # PaddleOCR results + raw/
│   └── glm_ocr/                            # GLM-OCR results + raw/
├── reports/                                # Evaluation reports (JSON + HTML)
├── docs/                                   # Project documentation
├── ground_truth.json                       # Expected extraction results
├── .env                                    # Azure credentials (gitignored)
└── pyproject.toml                          # Poetry dependencies
```
