# Databricks NLP Document Understanding Tool - Technical Implementation Plan

**Date:** 2026-04-07
**Audience:** Technical Leadership + Engineering
**Status:** Implementation-Ready Draft

---

## Table of Contents

1. [Executive Recommendation](#section-1-executive-recommendation)
2. [Capability Mapping](#section-2-capability-mapping)
3. [Workspace-First Development Architecture](#section-3-workspace-first-development-architecture)
4. [Recommended Folder Structure](#section-4-recommended-folder-structure)
5. [Notebook and File Connectivity Instructions](#section-5-notebook-and-file-connectivity-instructions)
6. [Extraction Pipeline Design](#section-6-extraction-pipeline-design)
7. [Databricks-Native Options](#section-7-databricks-native-options)
8. [Agent Strategy](#section-8-agent-strategy)
9. [Experiment and Evaluation Strategy](#section-9-experiment-and-evaluation-strategy)
10. [Phased Delivery Plan](#section-10-phased-delivery-plan)
11. [Production Evolution Path](#section-11-production-evolution-path)
12. [Code Starter Examples](#section-12-code-starter-examples)
13. [Final Recommendation](#section-13-final-recommendation)

---

# Section 1: Executive Recommendation

## Recommended Architecture: Hybrid -- Custom Extraction Pipeline + Genie for Structured Query Layer

**Decision:** Neither Genie-only nor custom-extraction-only is sufficient. The correct architecture is a two-tier hybrid:

1. **Tier 1 -- Custom Document Parsing and Extraction Pipeline** (primary workload): Databricks Document Parsing + Information Extraction to ingest PDFs, DOCX, and spreadsheets from workspace-hosted blobs, convert them to structured Delta tables in Unity Catalog, and build a vector index over extracted content for Q&A.

2. **Tier 2 -- Genie as the Natural-Language Query Interface** (downstream only): Once extracted content lands in Unity Catalog tables/views, Genie provides natural-language querying over that structured data via a Pro or Serverless SQL Warehouse. Genie does not touch raw files.

**Why not Genie-only:**
Genie operates over Unity Catalog tables and views. It cannot open a PDF, parse a DOCX, or extract text from a spreadsheet blob sitting in a workspace volume. Attempting to force Genie into this role would require all upstream parsing to already be complete and materialized -- which is the harder problem. Genie is a query accelerator, not a document processor.

**Why not custom-extraction-only:**
Building a bespoke natural-language-to-SQL or natural-language-to-table-lookup interface when Genie already does this well is unnecessary engineering. Once structured tables exist, Genie adds immediate value with zero custom code for ad-hoc querying by business users.

**Why hybrid:**
The hybrid approach plays to each component's strength. Document Parsing and Information Extraction handle the unstructured-to-structured conversion. Vector Search enables semantic Q&A over content that does not reduce cleanly to tabular rows. Genie handles "show me all invoices over $10K from Q3" queries against the resulting tables. The Mosaic AI Agent Framework ties these capabilities together with a retrieval-augmented agent that routes questions to the appropriate backend.

**Critical prerequisite:** Raw blobs must be migrated from workspace storage into Unity Catalog Volumes before Document Parsing and Information Extraction can operate on them. This is a blocking step and should be the first engineering task.

---

# Section 2: Capability Mapping

| Business Need | Databricks Capability | Component | Notes |
|---|---|---|---|
| **Natural-language querying over structured data** | **Genie** (AI/BI) | Pro or Serverless SQL Warehouse + Unity Catalog tables/views | Only applicable after extraction pipeline has materialized structured tables. Genie queries tables, not blobs. Requires a Genie Space configured with relevant tables and trusted instructions. |
| **Unstructured document parsing (PDF, DOCX, images)** | **Document Parsing** | Serverless or job cluster via `databricks-document-parsing` library | Converts raw documents into structured markdown/text chunks. Supports PDF, DOCX, PPTX, images. Input files must reside in Unity Catalog Volumes. Output is structured content suitable for downstream extraction or direct vectorization. |
| **Schema-based information extraction** | **Information Extraction** | `databricks-information-extraction` library, serverless compute | Pulls typed fields (e.g., invoice number, date, vendor name) from parsed text using a user-defined schema. Requires Unity Catalog-enabled inputs. Currently public preview -- plan for API surface changes. |
| **Question answering over extracted content** | **Vector Search + Foundation Model Serving + Agent Framework** | Databricks Vector Search index on Delta table, FM endpoint (DBRX, Llama 3, or external), Agent tool | Embed parsed document chunks, store in Vector Search index, retrieve at query time, pass to LLM with context. This is the RAG path. |
| **Spreadsheet ingestion** | **Spark DataFrame reader (CSV/Excel) + Delta Lake** | Job cluster or serverless notebook | Use `spark.read.format("csv")` or the `com.crealytics.spark.excel` library for `.xlsx`. Parse into Delta tables in Unity Catalog. No special Databricks AI feature needed -- this is standard ETL. |
| **Experimentation and evaluation** | **MLflow Experiments + Mosaic AI Evaluation** | MLflow tracking server (workspace-managed) | Log agent responses, retrieval quality, and extraction accuracy as MLflow runs. Use `mlflow.evaluate()` with LLM judge metrics (faithfulness, relevance, chunk precision). |
| **Deployment and monitoring** | **Model Serving + Agent Framework + Inference Tables** | Serverless model serving endpoint | Deploy the agent as a serving endpoint via `mlflow.models.set_model()`. Inference tables automatically capture request/response payloads. Agent tracing logs to the associated MLflow experiment. |

---

# Section 3: Workspace-First Development Architecture

## 3.1 Design Philosophy

All project assets -- notebooks, Python modules, configuration files, and sample documents -- live together in a single Databricks Workspace folder (e.g., `/Workspace/Users/<user>/nlp-doc-understanding/` or `/Workspace/Repos/<user>/nlp-doc-understanding/`). This "colocated workspace-first" model means every team member who opens any notebook in the project can immediately resolve references to shared code and sample data without configuring external paths, volumes, or environment variables.

This architecture is explicitly a **prototype and development-phase design**. It is not the long-term production data architecture. The plan calls out where production boundaries will diverge.

## 3.2 Asset Classification: Notebooks vs. Python Modules

| Asset Type | Format | Rationale |
|---|---|---|
| Orchestration workflows (ingest, process, evaluate) | `.py` notebook (Databricks notebook format) | Notebooks provide cell-by-cell execution, visualization, `%run` chaining, and MLflow integration. These are the entry points. |
| Reusable parsing/extraction logic | `.py` module (plain Python file) | Must be importable. Functions like `extract_text_from_pdf()`, `chunk_document()`, `call_llm()` belong here. Notebooks cannot be imported as modules. |
| Configuration and prompts | `.yaml` or `.json` files | Separates tunable parameters (model names, chunk sizes, prompt templates) from code. Avoids hardcoding. |
| Sample documents | `.pdf`, `.docx`, `.xlsx`, `.txt` | Colocated in a `sample_data/` folder for immediate driver-local reads. |
| Evaluation datasets | `.csv` or `.jsonl` | Ground-truth labels and expected outputs for eval harness. |

**Key rule**: If the code defines reusable functions or classes that other files need to `import`, it must be a `.py` module, not a notebook. Databricks notebooks cannot be imported via Python's `import` statement. They can only be called via `%run`, which executes them in the caller's namespace (no scoping, no return values as module attributes).

## 3.3 How Notebooks Should Call Shared Code

There are exactly two mechanisms, and the project should use both for different purposes.

### Mechanism A: `%run` for Notebook-to-Notebook Setup

Use `%run` only for lightweight orchestration -- e.g., running a shared setup notebook that initializes Spark configs, sets common variables, or installs libraries.

```python
%run ./notebooks/00_setup
```

`%run` executes the target notebook **in the caller's namespace**. Every variable, function, and import defined in `00_setup` becomes available in the calling notebook. This is powerful but dangerous: name collisions are silent, and there is no encapsulation.

**Use `%run` only for**: setup/config notebooks that establish shared state. Never use it to "import" reusable logic. That is what `.py` modules are for.

### Mechanism B: Standard Python `import` for Reusable Modules

On DBR 11.0+ and all newer runtimes (including DBR 14.x LTS, 15.x, and serverless), Databricks adds the notebook's directory to `sys.path` automatically. This means `.py` files colocated in the workspace can be imported directly.

```python
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), '..', 'src'))

from parsing import extract_text_from_pdf
from chunking import chunk_document
```

The explicit `sys.path.insert` is the **recommended defensive pattern** because it works regardless of whether `os.getcwd()` points to the notebook's own folder or the project root. It is one line of boilerplate that eliminates an entire class of "ModuleNotFoundError" issues.

## 3.4 File Path Resolution

### The `os.getcwd()` Behavior

On newer Databricks runtimes (DBR 13.3+), when a notebook is stored as a workspace file, `os.getcwd()` returns the directory containing that notebook. For example:

- Notebook location: `/Workspace/Users/me/nlp-doc-understanding/notebooks/01_ingest.py`
- `os.getcwd()` returns: `/Workspace/Users/me/nlp-doc-understanding/notebooks`

This is the foundation for relative path resolution. All file reads in the project should be anchored to this.

### Canonical Path Resolution Pattern

Every notebook should use this pattern (defined once in the setup notebook and inherited via `%run`):

```python
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "sample_data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
```

All downstream code references `SAMPLE_DATA_DIR`, `CONFIG_DIR`, etc. -- never hardcoded absolute paths.

### How Sample PDFs/Docs/Sheets Should Be Loaded

For **driver-local reads** (the primary pattern during development):

```python
import os

pdf_path = os.path.join(SAMPLE_DATA_DIR, "contract_sample.pdf")

with open(pdf_path, "rb") as f:
    raw_bytes = f.read()

import pdfplumber
with pdfplumber.open(pdf_path) as pdf:
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
```

This works because Databricks mounts workspace files into the driver node's local filesystem under the `/Workspace/...` path. Standard Python `open()`, `pathlib.Path.read_bytes()`, and any library that reads from the filesystem (PyMuPDF, pdfplumber, python-docx, openpyxl) will work without modification.

## 3.5 Driver-Local Reads vs. Spark Reads of Workspace Files

This distinction is critical and misunderstanding it is the single most common source of runtime errors in workspace-file-based projects.

### Driver-Local Reads (Python `open()`, `pathlib`, any Python library)

- Execute on the **driver node only**.
- The `/Workspace/...` filesystem is a FUSE mount on the driver.
- Works reliably for all file sizes relevant to this prototype.
- This is the correct pattern for: PDF parsing, DOCX parsing, XLSX parsing, reading config files, reading prompt templates.
- **All NLP document processing in this project should use driver-local reads.** Documents are parsed one at a time (or in small batches) on the driver. There is no benefit to distributing single-document parsing across Spark executors.

### Spark Reads (`spark.read`, `dbutils.fs`, DataFrame API)

- Execute on **executor nodes** (workers), not just the driver.
- Executor nodes do **not** have the `/Workspace/...` FUSE mount.
- `spark.read.text("file:/Workspace/...")` may work on classic (non-serverless) compute because the `file:` scheme forces a driver-local read, but this is fragile and not recommended.
- `spark.read.text("/Workspace/...")` without the `file:` prefix will fail because Spark interprets it as a DBFS path.

**Bottom line for this project**: Do not use Spark to read workspace files. Read documents with Python on the driver. If you need Spark DataFrames (e.g., for storing extracted text, embeddings, or evaluation results), construct them in Python and convert:

```python
import pandas as pd

results = [{"doc_id": "contract_01", "text": extracted_text, "num_pages": 5}]
df = spark.createDataFrame(pd.DataFrame(results))
```

## 3.6 Serverless Compute Limitations

Databricks serverless compute introduces specific constraints:

1. **No workspace FUSE mount on executors.** Serverless executor nodes cannot access `/Workspace/...` paths. Any code that runs on executors (inside Spark UDFs, `mapPartitions`, or distributed `pandas_udf`) cannot read workspace files. This is a hard limitation, not a configuration issue.

2. **Driver-local reads still work.** The serverless driver node does mount workspace files. So the primary pattern for this project (Python reads on the driver) is unaffected. The prototype will function on serverless compute as long as all document I/O stays on the driver.

3. **`%run` works on serverless.** Notebook-to-notebook `%run` references work.

4. **Python module imports work on serverless.** Importing `.py` files from the workspace works because imports execute on the driver.

5. **Library installation differences.** On serverless, `%pip install` works but the available system libraries may differ from classic compute. Libraries like `pdfplumber`, `python-docx`, and `openpyxl` install fine. Libraries requiring C compilation or system-level dependencies (e.g., Tesseract OCR) may not be available on serverless. The project should test library availability early.

6. **Production migration path.** When this prototype moves to production and processes documents at scale, files must move out of workspace files and into Unity Catalog Volumes (`/Volumes/<catalog>/<schema>/<volume>/`) or cloud object storage (S3/ADLS/GCS). Volumes are accessible from both driver and executors, support Spark reads, and integrate with Unity Catalog governance. This migration is called out here as a future requirement, not a current blocker.

---

# Section 4: Recommended Folder Structure

```
nlp-doc-understanding/
|
+-- notebooks/
|   +-- 00_setup.py                  # Shared setup: path constants, library installs,
|   |                                #   Spark configs, common imports
|   +-- 01_ingest_and_parse.py       # Load sample docs, extract raw text, store results
|   +-- 02_chunk_and_embed.py        # Chunk extracted text, generate embeddings
|   +-- 03_llm_extraction.py         # Send chunks to LLM for entity/field extraction
|   +-- 04_postprocess.py            # Clean, validate, and structure LLM outputs
|   +-- 05_evaluate.py               # Run eval harness against ground truth
|   +-- 06_demo_end_to_end.py        # Single-click demo: ingest -> extract -> display
|
+-- src/
|   +-- __init__.py                  # Makes src/ a Python package (can be empty)
|   +-- parsing.py                   # extract_text_from_pdf(), extract_text_from_docx(),
|   |                                #   extract_text_from_xlsx(). Pure Python, no Spark.
|   +-- chunking.py                  # chunk_document(), sliding_window_chunk().
|   |                                #   Text splitting strategies with configurable
|   |                                #   chunk size and overlap.
|   +-- embeddings.py                # get_embeddings(). Wraps calls to Databricks
|   |                                #   Foundation Model API or external embedding endpoints.
|   +-- llm.py                       # call_llm(), build_prompt(). Wraps calls to
|   |                                #   Databricks Model Serving or external LLM APIs.
|   |                                #   Handles retries, rate limiting, response parsing.
|   +-- postprocess.py               # validate_extraction(), normalize_fields().
|   |                                #   Cleans and structures raw LLM outputs.
|   +-- eval_utils.py                # compute_metrics(), compare_to_ground_truth().
|   |                                #   Evaluation logic separated from the eval notebook.
|   +-- path_utils.py                # resolve_project_root(), get_sample_data_dir().
|                                    #   Centralized path resolution.
|
+-- sample_data/
|   +-- pdfs/
|   |   +-- contract_sample_01.pdf
|   |   +-- invoice_sample_01.pdf
|   |   +-- report_sample_01.pdf
|   +-- docx/
|   |   +-- memo_sample_01.docx
|   +-- xlsx/
|   |   +-- financials_sample_01.xlsx
|   +-- txt/
|       +-- plaintext_sample_01.txt
|
+-- configs/
|   +-- model_config.yaml            # Model serving endpoint names, temperature,
|   |                                #   max_tokens, model version identifiers.
|   +-- chunking_config.yaml         # chunk_size, overlap, splitting strategy.
|   +-- prompts/
|   |   +-- entity_extraction.txt    # Prompt template for entity extraction.
|   |   +-- summarization.txt        # Prompt template for doc summarization.
|   |   +-- field_extraction.txt     # Prompt template for structured field extraction.
|   +-- eval_config.yaml             # Thresholds, metric definitions, pass/fail criteria.
|
+-- eval/
|   +-- ground_truth/
|   |   +-- contract_sample_01.json  # Expected extraction output for each sample doc.
|   |   +-- invoice_sample_01.json
|   |   +-- report_sample_01.json
|   +-- results/                     # Gitignored. Eval notebooks write here.
|       +-- .gitkeep
|
+-- output/
|   +-- .gitkeep                     # Gitignored. Notebooks write intermediate and
|                                    #   final outputs here during development.
|
+-- deployment/
|   +-- cluster_init.sh              # Init script for library/system deps
|   +-- workflow_config.json         # Databricks Workflow job definitions
|   +-- serving_config.json          # Model serving endpoint configuration
|
+-- README.md
```

### Folder-by-Folder Rationale

| Folder | Purpose | Key Rules |
|---|---|---|
| `notebooks/` | All executable Databricks notebooks. Numbered for execution order. | No reusable functions defined here. Notebooks orchestrate; `src/` implements. |
| `src/` | All reusable Python modules. Imported by notebooks. | Must include `__init__.py`. No notebook-format files. No Spark session creation (receive `spark` from caller). |
| `sample_data/` | Small sample documents for development and testing. | Keep files under 10 MB each. Subdirectories by file type. Not for production data -- production documents go to UC Volumes. |
| `configs/` | YAML/JSON configuration and prompt templates. | Loaded with `yaml.safe_load()` or `json.load()` on the driver. Never hardcode values that belong here into notebooks or src. |
| `eval/` | Ground truth files and evaluation results. | `ground_truth/` is version-controlled. `results/` is gitignored (generated output). |
| `output/` | Scratch space for notebook outputs during development. | Gitignored. Notebooks may write CSVs, JSONs, or Parquet here for inspection. Not a production output location. |
| `deployment/` | Init scripts, workflow configs, serving configs. | Production deployment artifacts. |

---

# Section 5: Notebook and File Connectivity Instructions

## 5.1 Notebook-to-Notebook References (`%run`)

Notebooks in the same folder reference each other with `./` relative paths. The path is relative to the calling notebook's location, not to `os.getcwd()`.

```python
%run ./00_setup
```

The `./` prefix is required. Without it, Databricks resolves the path relative to the user's home folder, not the notebook's folder -- a common source of "Notebook not found" errors.

**Do not include the `.py` extension in `%run` paths.** Databricks resolves the notebook name without extension.

## 5.2 Importing Python Modules from `src/`

Every notebook that needs `src/` modules should execute this pattern, typically inherited from `00_setup` via `%run`:

```python
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
```

After this, all notebooks that `%run ./00_setup` can do:

```python
from parsing import extract_text_from_pdf
from chunking import chunk_document
from llm import call_llm, build_prompt
```

## 5.3 Relative Path Handling for Data and Config Files

All file reads should go through the centralized path constants defined in `00_setup.py` or `src/path_utils.py`.

```python
import os

def resolve_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_sample_data_dir():
    return os.path.join(resolve_project_root(), "sample_data")

def get_config_dir():
    return os.path.join(resolve_project_root(), "configs")

def get_output_dir():
    return os.path.join(resolve_project_root(), "output")
```

When called from a `.py` module, `os.path.dirname(__file__)` reliably returns the module's directory. When called from a notebook, `__file__` is not defined -- notebooks should use `os.getcwd()` and the `PROJECT_ROOT` constant from `00_setup`.

**Rule**: Modules in `src/` use `__file__`. Notebooks use `os.getcwd()`. The `00_setup` notebook bridges the two.

## 5.4 DO THIS / AVOID THIS

| Scenario | DO THIS | AVOID THIS | Why |
|---|---|---|---|
| **Notebook calls another notebook** | `%run ./00_setup` | `%run 00_setup` or `%run /Users/me/project/notebooks/00_setup` | Without `./`, Databricks resolves relative to user home. Absolute paths break portability. |
| **Notebook imports a .py module** | `sys.path.insert(0, ...)` then `from parsing import func` | `%run ./src/parsing` | `%run` executes a notebook, not a module. It dumps all names into the caller's namespace with no scoping. |
| **Reading a workspace file in a notebook** | `open(os.path.join(SAMPLE_DATA_DIR, "pdfs", "file.pdf"), "rb")` | `dbutils.fs.cp("file:/Workspace/...", "/tmp/file.pdf")` followed by local read | The double-copy pattern is unnecessary. Workspace files are already on the driver's local filesystem. |
| **Reading a workspace file in a Spark UDF** | Move the file to a UC Volume first. Pass the Volume path to the UDF. | `open("/Workspace/.../file.pdf")` inside a UDF | Executors do not have the workspace FUSE mount. |
| **Resolving project root in a .py module** | `os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))` | `os.getcwd()` | `os.getcwd()` in a module reflects the calling notebook's directory, not the module's. |
| **Resolving project root in a notebook** | `os.path.abspath(os.path.join(os.getcwd(), ".."))` | `os.path.dirname(__file__)` | `__file__` is not defined in Databricks notebooks. |
| **Specifying file extension in %run** | `%run ./00_setup` | `%run ./00_setup.py` | Databricks `%run` resolves notebook names without extensions. |
| **Sharing constants across notebooks** | Define in `00_setup.py`, propagate via `%run ./00_setup` | Define in each notebook separately | Duplicated definitions drift. |
| **Installing libraries** | `%pip install pdfplumber python-docx openpyxl` in `00_setup.py` | `!pip install ...` or installing in a later notebook | `%pip` restarts the Python interpreter cleanly. `!pip` does not. |
| **Hardcoding a user-specific path** | Use relative resolution from `os.getcwd()` or `__file__` | `/Workspace/Users/john.doe@company.com/...` | Absolute user paths break when another team member clones the project. |
| **Passing file paths to src/ functions** | Pass the fully resolved path as a string argument | Have the function internally construct paths | Functions in `src/` should be path-agnostic. The caller resolves paths. |
| **Using serverless compute** | Keep all file I/O on the driver. Parse documents in regular Python. | Distributing document reads across executors | Serverless executors cannot access workspace files. |

## 5.5 Module Reload Caveats During Development

When iterating on `src/` modules during development, Python caches imports. Two solutions:

```python
import importlib
import parsing
importlib.reload(parsing)
from parsing import extract_text_from_pdf
```

Or use autoreload magic (preferred during interactive development):

```python
%load_ext autoreload
%autoreload 2
```

Place `%load_ext autoreload` and `%autoreload 2` in `00_setup.py` so it applies to all notebooks during development. Remove or disable before production deployment.

---

# Section 6: Extraction Pipeline Design

## 6.1 Architecture Overview

The extraction pipeline follows a driver-node-only execution model for Phase 1. This is a deliberate architectural choice driven by two constraints: (1) serverless compute executors cannot read workspace files via `/Workspace/` paths, and (2) Python parsing libraries like PyMuPDF and pdfplumber are not serializable across Spark executors without custom setup. All file parsing runs single-threaded on the driver using Python libraries. Parallelism is achieved by distributing files across multiple concurrent notebook jobs via Databricks Workflows, not via Spark parallelism within a single job.

The pipeline has seven stages:

```
[Workspace Files] --> (1) MIME Detection --> (2) Router --> (3) Parser --> (4) Normalized Document --> (5) Quality Scoring --> (6) Chunking --> (7) Delta Table Write
```

## 6.2 Runtime and Library Specifications

**Target Runtime**: Databricks Runtime 15.4 LTS ML or 16.x

**Pre-installed on DBR 15.4 LTS** (no `%pip install` required):
- `pandas` (2.1.x+)
- `chardet`
- `openpyxl`

**Libraries requiring installation**:

| Library | Version | Purpose | Install Command |
|---------|---------|---------|-----------------|
| `PyMuPDF` | `1.24.3+` | PDF text/table extraction | `%pip install PyMuPDF==1.24.3` |
| `pdfplumber` | `0.11.0+` | PDF table extraction (superior table detection) | `%pip install pdfplumber==0.11.0` |
| `pytesseract` | `0.3.10` | OCR for scanned PDFs | `%pip install pytesseract==0.3.10` |
| `python-docx` | `1.1.0+` | DOCX parsing | `%pip install python-docx==1.1.0` |
| `python-magic` | `0.4.27` | MIME type detection | `%pip install python-magic==0.4.27` |
| `xlrd` | `2.0.1` | Legacy `.xls` reading | `%pip install xlrd==2.0.1` |
| `tiktoken` | `0.7.0+` | Token-aware chunking | `%pip install tiktoken==0.7.0` |

**Tesseract system dependency**: OCR requires the `tesseract-ocr` binary. Install via init script:

```bash
#!/bin/bash
apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng
```

On serverless, Tesseract is not available -- OCR is deferred to Phase 2 with Databricks Document Parsing.

## 6.3 Normalized Intermediate Schema

All parsers output to this common schema:

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


@dataclass
class ExtractedTable:
    table_id: str
    page_number: Optional[int]
    sheet_name: Optional[str]
    headers: list[str]
    rows: list[list[str]]
    row_count: int
    col_count: int
    extraction_method: str
    confidence: float


@dataclass
class ExtractionError:
    stage: str
    severity: str
    message: str
    exception_type: Optional[str] = None
    page_number: Optional[int] = None


@dataclass
class ParsedDocument:
    doc_id: str
    source_path: str
    file_name: str
    file_type: str
    file_extension: str
    file_size_bytes: int
    extracted_text: str
    pages: list[dict]
    page_count: int
    tables: list[ExtractedTable]
    table_count: int
    metadata: dict
    extraction_timestamp: str
    extraction_duration_ms: int
    extraction_method: str
    extraction_quality_score: float
    ocr_applied: bool
    encoding_detected: Optional[str]
    errors: list[ExtractionError]
    is_successful: bool


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    total_chunks: int
    text: str
    token_count: int
    char_count: int
    start_page: Optional[int]
    end_page: Optional[int]
    source_path: str
    file_name: str
    file_type: str
    metadata: dict
    contains_table: bool
    extraction_timestamp: str
```

## 6.4 MIME Detection and File Type Router

```python
import magic
import os
from pathlib import Path

PARSER_ROUTES = {
    "application/pdf": "pdf_parser",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx_parser",
    "application/msword": "doc_legacy_parser",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx_parser",
    "application/vnd.ms-excel": "xls_parser",
    "text/csv": "csv_parser",
    "text/plain": "text_parser",
    "application/csv": "csv_parser",
}

EXTENSION_OVERRIDES = {
    ".csv": "csv_parser",
    ".tsv": "csv_parser",
    ".xls": "xls_parser",
    ".xlsx": "xlsx_parser",
    ".doc": "doc_legacy_parser",
    ".docx": "docx_parser",
    ".pdf": "pdf_parser",
    ".txt": "text_parser",
    ".md": "text_parser",
    ".log": "text_parser",
    ".json": "text_parser",
    ".xml": "text_parser",
}


def detect_file_type(file_path: str) -> tuple[str, str]:
    mime_type = magic.from_file(file_path, mime=True)
    extension = Path(file_path).suffix.lower()

    if mime_type == "text/plain" and extension in (".csv", ".tsv"):
        return mime_type, "csv_parser"

    parser = PARSER_ROUTES.get(mime_type)
    if parser:
        return mime_type, parser

    parser = EXTENSION_OVERRIDES.get(extension)
    if parser:
        return mime_type, parser

    return mime_type, "unsupported"
```

## 6.5 File-Type-Specific Parser Strategies

### PDF Strategy
- **Primary**: PyMuPDF (`fitz`) for text extraction -- fastest C-backed Python PDF library
- **Tables**: `pdfplumber` for table detection via line intersection algorithm
- **OCR fallback**: `pytesseract` when PyMuPDF returns <50 chars/page (scanned document heuristic)
- **Phase 2 upgrade**: Replace with Databricks Document Parsing (`ai_parse_document()`) once files are in UC Volumes

### DOCX Strategy
- **Primary**: `python-docx` for paragraphs, tables, headers/footers, core properties
- **Legacy .doc**: `antiword` for quick text extraction; `libreoffice --headless --convert-to docx` as fallback with full table support

### Spreadsheet Strategy (CSV/XLS/XLSX)
- **CSV**: `chardet` for encoding detection, `csv.Sniffer` for delimiter detection, `pandas.read_csv` with `dtype=str`
- **XLSX**: `pandas.read_excel` with `openpyxl` engine; metadata from `openpyxl.load_workbook`
- **XLS**: `pandas.read_excel` with `xlrd` engine
- **All columns read as `dtype=str`** to avoid type coercion errors on mixed-type columns

### Text File Strategy
- `chardet` for encoding detection from first 64KB
- Hard fallback to `latin-1` (never raises decoding errors)
- Line count and character count in metadata

### Fallback Chain Per File Type

| Primary Parser | Fallback 1 | Fallback 2 | Final |
|---|---|---|---|
| `pymupdf+pdfplumber` (PDF) | `pymupdf` text-only | `pdfplumber` text-only | `_failed_document` |
| `python-docx` (DOCX) | `libreoffice --convert-to txt` | Binary heuristic extraction | `_failed_document` |
| `antiword` (DOC) | `libreoffice --convert-to docx` | `libreoffice --convert-to txt` | `_failed_document` |
| `pandas+openpyxl` (XLSX) | `openpyxl` direct | `_failed_document` | -- |
| `chardet+read` (TXT) | `latin-1` read | `_failed_document` | -- |

## 6.6 Extraction Quality Scoring

Composite heuristic (0.0 to 1.0):

| Component | Weight | Logic |
|---|---|---|
| Text density | 0.4 | Chars/page relative to expected 200-5000 range |
| Error penalty | 0.2 | Deduction per error (0.3) and warning (0.1) |
| Table integrity | 0.2 | Average table confidence (fill rate + column consistency) |
| Metadata completeness | 0.1 | Fraction of key fields populated |
| Encoding confidence | 0.1 | `chardet` confidence for text files |

## 6.7 Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | 512 tokens | Optimized for embedding models (BGE, E5, GTE) 512-token sweet spots |
| `chunk_overlap` | 64 tokens (12.5%) | ~2-3 sentences at boundaries captured in both chunks |
| `tokenizer` | `tiktoken` with `cl100k_base` | Compatible with most LLMs on Databricks |
| `min_chunk_size` | 50 tokens | Discard trailing whitespace/footer chunks |
| `chunk_id` | `SHA-256(doc_id + ":" + chunk_index)` | Deterministic, enables idempotent upserts |

**Tables get separate chunks**: Sliding-window chunking over narrative text would split table rows arbitrarily. Tables are converted to Markdown format and chunked separately, with the header row repeated in each chunk for self-contained interpretability.

## 6.8 Parallelism Strategy

- **Small workloads (up to ~1000 files)**: Python `concurrent.futures.ThreadPoolExecutor` on the driver, `max_workers=4`
- **Large workloads (1000+ files)**: Databricks Workflows fan-out -- partition input into batches, submit each as a separate notebook task with `max_concurrency`

## 6.9 Delta Table Schema

**Table 1: `catalog.schema.parsed_documents`** -- Full parsed output per document, partitioned by `file_extension`, Z-ORDER on `extraction_timestamp`.

**Table 2: `catalog.schema.document_chunks`** -- Chunked text with metadata, partitioned by `file_type`, Z-ORDER on `doc_id`. Source table for Vector Search Delta Sync index.

---

# Section 7: Databricks-Native Options

## 7.1 Genie (AI/BI)

**Should do**: Natural-language interface for business users querying structured extraction results over curated Unity Catalog views. Use trusted instructions to constrain scope.

**Should NOT do**: Parse raw files. Answer open-ended questions about content not in table columns. Replace the RAG Q&A path.

**Requires**: Pro SQL Warehouse or Serverless SQL Warehouse. Classic SQL warehouses do not support Genie.

## 7.2 Document Parsing

**Should do**: Convert raw PDFs, DOCX, PPTX, images into structured text/markdown. Run as batch pipeline on files in UC Volumes. Handle OCR automatically.

**Should NOT do**: Extract typed business fields (that is Information Extraction). Serve as real-time API. Process files still in workspace DBFS.

## 7.3 Information Extraction

**Should do**: Extract schema-defined fields from parsed text. Materialize into strongly-typed Delta tables for Genie. Support iterative schema refinement.

**Should NOT do**: Replace general-purpose Q&A. Run on files not in Unity Catalog. Serve as sole extraction for spreadsheets.

**Note**: Public preview. Pin library versions. Test against fixed document sets before scaling.

## 7.4 MLflow Experiments

**Should do**: Track every iteration of extraction schemas, prompts, chunking strategies. Store evaluation metrics. A/B test configurations. Receive agent traces automatically.

**Should NOT do**: Act as production monitoring dashboard (use inference tables + SQL dashboards). Store raw document files.

## 7.5 Mosaic AI Agent Framework

**Should do**: Define the Q&A agent with tool-calling. Provide retriever/SQL/lookup tools. Enable serving endpoint deployment. Support review app for human eval.

**Should NOT do**: Replace the extraction pipeline. Manage ETL scheduling.

## 7.6 Tracing

**Should do**: Instrument agent calls with MLflow spans. Auto-log for deployed agents. Debug retrieval failures.

**Should NOT do**: Trace the batch extraction pipeline (use Spark UI/job logs for that).

## 7.7 Evaluation

**Should do**: `mlflow.evaluate()` with LLM judge metrics. Gate deployments on threshold scores (faithfulness > 0.85, relevance > 0.80).

**Should NOT do**: Replace domain-expert review. Evaluate Genie query quality (monitor separately).

## 7.8 Vector Search

**Should do**: Delta Sync index over document chunks. Databricks-hosted embedding model. Retriever tool for the Q&A agent.

**Should NOT do**: Index structured fields (use SQL/Genie). Replace keyword search for exact-match queries.

---

# Section 8: Agent Strategy

## 8.1 Phase 1: Single Agent with Multiple Tools

**Decision:** Deploy a single Mosaic AI Agent Framework agent with three tools. Do not build multiple agents in phase 1.

**Agent name:** `document_qa_agent`

**Foundation model:** `databricks-meta-llama-3-1-70b-instruct` on Model Serving. Upgrade only if evaluation shows insufficient reasoning quality.

## 8.2 Tool Definitions

### Tool 1: `semantic_retriever`
- **Purpose:** Retrieve relevant document passages for open-ended/semantic questions
- **Implementation:** Databricks Vector Search retriever tool
- **When to call:** "What does the contract say about termination?", "Summarize the key findings"
- **Boundary:** Not for structured field lookups

### Tool 2: `structured_lookup`
- **Purpose:** Query extracted structured fields via parameterized SQL
- **When to call:** "What is the invoice total?", "How many contracts expire before 2025?"
- **Boundary:** SELECT only. Validate against table/column allowlist.

### Tool 3: `document_metadata_lookup`
- **Purpose:** Retrieve document metadata (names, types, dates, page counts)
- **When to call:** "What documents do we have?", "Which PDFs were uploaded last week?"
- **Boundary:** Metadata only, no content

## 8.3 Routing Logic

1. **Specific field values** -> `structured_lookup`
2. **Document content/meaning/interpretation** -> `semantic_retriever`
3. **What documents exist** -> `document_metadata_lookup`
4. **Both fields and passages** -> `structured_lookup` first, then `semantic_retriever`
5. **Out of scope** -> Decline to answer

## 8.4 Hallucination Prevention (Non-Negotiable)

1. **Grounding mandate**: Answer ONLY from retrieved context. Decline if insufficient.
2. **Source citation requirement**: Every answer must cite file name + page number.
3. **Empty retrieval handling**: Zero chunks above 0.3 cosine similarity -> decline.
4. **Tool-call enforcement**: Must call at least one tool before answering.
5. **Deployment gate**: Faithfulness > 0.85 on golden eval set or do not deploy.
6. **SQL guardrails**: Validate generated SQL against table/column allowlist.

## 8.5 Phase 2 Evolution (Not Phase 1)

- Split into router + specialist agents if single agent hits limits
- Add document onboarding agent for operational tasks
- Integrate Genie as a tool when API-based invocation is available

---

# Section 9: Experiment and Evaluation Strategy

## 9.1 Workspace Experiments vs. Notebook Experiments

**Notebook Experiments**: Auto-created when `mlflow.start_run()` is called without an active experiment. Bound to that notebook path. Runs only visible from that notebook. If moved or deleted, association breaks.

**Workspace Experiments**: Explicitly created at a workspace path. Any notebook, job, or agent can log to it. Multiple engineers share the same experiment.

**Recommendation:** Use workspace experiments exclusively. Never rely on implicit notebook experiments.

## 9.2 Experiment Naming Conventions

```
/Workspace/Users/<project-owner>/doc-understanding/
  experiments/
    extraction-eval
    retrieval-eval
    agent-eval
    latency-benchmarks
    cost-tracking
```

**Run naming:** `{component}-{dataset}-{timestamp}` (e.g., `extraction-invoices-20260407T1430`)

**Run tags:**

| Tag Key | Example Value | Purpose |
|---|---|---|
| `doc_type` | `invoice`, `contract` | Filter by document class |
| `model_version` | `dbrx-instruct-v1` | Track foundation model |
| `pipeline_stage` | `extraction`, `retrieval`, `agent` | Identify component |
| `dataset_version` | `gold-set-v3` | Link to eval dataset |
| `environment` | `dev`, `staging`, `prod` | Deployment stage |

## 9.3 Evaluation Matrix

| Metric | What It Measures | Target | How to Compute |
|---|---|---|---|
| **Extraction Completeness** | % of expected fields extracted | >= 95% | `len(extracted & expected) / len(expected)` against gold schema |
| **Field Accuracy** | Correctness of extracted values | >= 90% exact, >= 95% fuzzy | Exact match for structured fields, Levenshtein >= 0.95 for text |
| **Table Preservation Quality** | Fidelity of tabular data | >= 85% cell-level F1 | Cell-level precision/recall against gold tables |
| **Citation Grounding** | Answers traceable to sources | >= 90% grounded | LLM judge via `mlflow.evaluate()` `groundedness` metric |
| **Chunk Relevance** | Retrieved chunks relevant to query | P@5 >= 0.7, NDCG@5 >= 0.75 | Precision/recall/NDCG against labeled passages |
| **Answer Correctness** | Final answer factually correct | >= 85% | `correctness` metric in Mosaic AI Agent Evaluation |
| **Answer Faithfulness** | Answer faithful to context (no hallucination) | >= 92% | `faithfulness` metric in Mosaic AI Agent Evaluation |
| **Latency (p50, p95)** | End-to-end response time | p50 <= 8s, p95 <= 20s | Wall-clock time per request from trace spans |
| **Failure Rate** | Errors, timeouts, malformed output | <= 2% | `failures / total_requests` |
| **Cost per Document** | Full pipeline cost for one document | <= $0.15 for 10-page PDF | Sum token usage across all LLM calls x per-token pricing |

## 9.4 Evaluation Dataset Management

- **`eval_extraction_gold`**: 50-100 documents with verified field extractions per doc type
- **`eval_retrieval_gold`**: 200+ query-passage pairs with binary relevance labels
- **`eval_qa_gold`**: 150+ question-answer pairs with source citations

Version with Delta table history or MLflow run tags. Never overwrite -- append new versions.

---

# Section 10: Phased Delivery Plan

## Phase 1: Workspace-Based Prototype (Weeks 1-3)

**Goals:** Prove documents can be ingested, parsed, chunked, and queried. Establish baseline metrics.

**Deliverables:**
1. Workspace folder structure created
2. Ingestion notebook: reads PDFs from workspace, parses with Python, writes to Delta
3. Chunking notebook: 512-token chunks with 64-token overlap
4. Embedding notebook: vectors via Foundation Model API
5. Simple retrieval notebook: brute-force cosine similarity
6. Workspace MLflow experiments with baseline runs
7. 10-20 curated test documents

**Risks:**
- PDF parsing quality varies by type (scanned vs. native, tables vs. prose)
- Foundation Model API rate limits or preview access delays
- Workspace file storage has size limits (files up to 500 MB)

**Dependencies:** DBR 14.3+ or 15.x LTS, Foundation Model API access, Unity Catalog enabled

**Exit Criteria:**
- Can ingest a 10-page PDF and retrieve relevant chunks in a single notebook session
- Extraction completeness and retrieval precision baselines logged to MLflow
- At least 3 document types tested

## Phase 2: Structured Extraction and Retrieval (Weeks 4-6)

**Goals:** Replace brute-force with Vector Search. Implement structured extraction. Store in Unity Catalog.

**Deliverables:**
1. Vector Search index (Delta Sync) on chunks table
2. Retrieval with metadata filters
3. Structured extraction via Foundation Model API with JSON-mode
4. Table extraction sub-pipeline
5. Gold-standard evaluation datasets (50+ labeled examples each)
6. Unity Catalog schema: `catalog.doc_understanding.{raw_documents, parsed_text, chunks, doc_extractions, doc_tables}`

**Risks:**
- Vector Search requires UC-managed source table (migration needed if Phase 1 used hive_metastore)
- Extraction quality depends on prompt engineering and JSON-mode reliability
- Information Extraction (native) is public preview -- may be unstable

**Exit Criteria:**
- Vector Search precision@5 >= 0.6
- Extraction completeness >= 85% on gold set
- All data in Unity Catalog with column comments and descriptions

## Phase 3: Agent and Evaluation Layer (Weeks 7-9)

**Goals:** Build Mosaic AI Agent. Implement systematic evaluation. Deploy to serving endpoint.

**Deliverables:**
1. Agent with semantic_retriever, structured_lookup, document_metadata_lookup tools
2. Registered in MLflow as pyfunc model
3. Deployed to Model Serving endpoint (CPU, scale-to-zero)
4. `mlflow.evaluate()` with correctness, faithfulness, groundedness
5. Gold QA dataset (100+ question-answer-context triples)
6. Trace logging confirmed
7. Basic SQL dashboard (latency, failure rate, query volume)

**Risks:**
- Multi-tool orchestration failure cascades
- LLM judge metrics depend on gold data quality
- Serving endpoint cold-start latency

**Exit Criteria:**
- Answer correctness >= 75% (targeting 85% by Phase 4)
- Faithfulness >= 85%, citation grounding >= 80%
- Responses within 20s (p95)
- Traces visible in MLflow experiment UI

## Phase 4: Production Hardening (Weeks 10-13)

**Goals:** Migrate to production architecture. Automate ingestion. Harden monitoring.

**Deliverables:**
1. Files migrated to UC Volumes or cloud object storage
2. Ingestion orchestrated via Databricks Workflows (5-task DAG)
3. Unity Catalog governance (column-level ACLs, row-level security)
4. Model Registry promotion gates (Staging -> Production)
5. Automated nightly evaluation pipeline
6. Production SQL dashboard with alerts
7. Genie space over extraction tables
8. All target metrics met

**Risks:**
- Workflow orchestration complexity
- UC governance changes may require admin coordination
- Cost at scale may exceed projections

**Exit Criteria:**
- Zero manual steps: document dropped in landing zone is queryable within 15 minutes
- All evaluation matrix targets met
- Monitoring live with verified alerting
- Genie space operational for business users

---

# Section 11: Production Evolution Path

## 11.1 File Storage Migration

| Storage Option | When to Use |
|---|---|
| **UC Volumes (Managed)** | Default. Documents within UC governance. Up to 5 GB per file. `/Volumes/catalog/schema/volume/` |
| **UC Volumes (External)** | Documents already in cloud storage. External location pointing to S3/ADLS/GCS. |
| **Cloud Object Storage (Direct)** | TB-scale volumes or external system writes. Requires IAM/storage credential. |

## 11.2 Ingestion Orchestration

**Recommended: Databricks Workflows** (batch/event-driven)

```
Workflow: doc_understanding_ingestion
+-- Task 1: detect_new_files (scheduled or event-triggered)
+-- Task 2: parse_and_chunk (depends on Task 1)
+-- Task 3: generate_embeddings (depends on Task 2)
+-- Task 4: extract_structured_fields (depends on Task 2)
+-- Task 5: update_vector_index (depends on Task 3)
```

Start with Workflows. Evaluate Delta Live Tables in Phase 4 if continuous ingestion becomes a requirement.

## 11.3 Access Control Evolution

**Tier 1 -- Schema Level:** Team-scoped access to `doc_understanding` schema
**Tier 2 -- Table Level:** Analysts read extractions but not raw text/chunks
**Tier 3 -- Column/Row Level:** PII masking functions, department-based row filters
**Service principal:** `sp-doc-understanding-prod` with least-privilege grants for all jobs and serving

## 11.4 When Genie Becomes Useful

**Not yet useful**: Phase 1-2 (data in workspace files or poorly structured tables, generic MAP columns)

**Becomes useful when:**
1. Extractions stored in well-modeled tables with explicit columns (not `MAP<STRING,STRING>`)
2. Tables have UC descriptions and column comments
3. Sufficient volume for analytical queries (hundreds+ of extracted documents)

**Activation (Phase 4):** Create flattened `doc_extractions_flat` table, add comprehensive comments, create Genie space with sample questions.

## 11.5 Transition Without Breaking Dev

- **Dual-path configuration**: `config.py` with `ENV` widget switching dev/prod paths
- **Repos for code promotion**: Move to Git-backed Repos by Phase 3. Production runs from `main` branch.
- **Schema mirroring**: `doc_understanding_dev` mirrors production schema
- **Experiment isolation**: Dev logs to user paths, production to `/Workspace/Shared/` paths

---

# Section 12: Code Starter Examples

## 12.1 Reading Workspace-Based Files

```python
import os
import pathlib


def resolve_workspace_path(relative_path: str) -> str:
    notebook_directory = os.getcwd()
    return str(pathlib.Path(notebook_directory) / relative_path)


def strip_file_scheme(workspace_file_uri: str) -> str:
    return workspace_file_uri.replace("file:", "")


pdf_path_via_cwd = resolve_workspace_path("data/documents/sample.pdf")
docx_path_via_cwd = resolve_workspace_path("data/documents/sample.docx")
xlsx_path_via_cwd = resolve_workspace_path("data/documents/sample.xlsx")

pdf_path_explicit = "file:/Workspace/Users/team@company.com/nlp-tool/data/documents/sample.pdf"

with open(pdf_path_via_cwd, "rb") as pdf_file_handle:
    pdf_binary_content = pdf_file_handle.read()

with open(strip_file_scheme(pdf_path_explicit), "rb") as pdf_explicit_handle:
    pdf_explicit_binary_content = pdf_explicit_handle.read()

print(f"PDF bytes (cwd): {len(pdf_binary_content)}")
print(f"PDF bytes (explicit): {len(pdf_explicit_binary_content)}")
```

## 12.2 Importing Local Modules

```python
import sys
import os
import pathlib


def register_source_directory(relative_source_path: str) -> None:
    notebook_directory = os.getcwd()
    absolute_source_path = str(
        pathlib.Path(notebook_directory) / relative_source_path
    )
    if absolute_source_path not in sys.path:
        sys.path.insert(0, absolute_source_path)


register_source_directory("../src")

from document_parser import parse_document
from extraction_utils import clean_extracted_text
from schema_models import ParsedDocument
```

For notebook-to-notebook imports (execute in a Databricks cell):

```python
%run ../src/document_parser_notebook
%run ../src/extraction_utils_notebook
```

## 12.3 Loading Binary Files for PDF Parsing

```python
import os
import pathlib
import fitz


def load_binary_content_from_workspace(relative_file_path: str) -> bytes:
    notebook_directory = os.getcwd()
    absolute_file_path = pathlib.Path(notebook_directory) / relative_file_path
    with open(absolute_file_path, "rb") as binary_file_handle:
        return binary_file_handle.read()


def open_pdf_from_workspace(relative_file_path: str) -> fitz.Document:
    pdf_binary_content = load_binary_content_from_workspace(relative_file_path)
    return fitz.open(stream=pdf_binary_content, filetype="pdf")


pdf_document = open_pdf_from_workspace("data/documents/sample.pdf")
total_page_count = pdf_document.page_count

all_page_text_blocks: list[str] = []
for page_index in range(total_page_count):
    page = pdf_document.load_page(page_index)
    all_page_text_blocks.append(page.get_text("text"))

full_document_text = "\n".join(all_page_text_blocks)
pdf_document.close()

print(f"Pages extracted: {total_page_count}")
print(f"Total characters: {len(full_document_text)}")
```

## 12.4 Parsing Files into a Standard Schema

```python
import os
import time
import pathlib
import pandas as pd
import fitz
from dataclasses import dataclass, field
from docx import Document as DocxDocument
from openpyxl import load_workbook


@dataclass
class ParsedDocument:
    source_file_path: str
    file_type: str
    raw_text: str
    page_count: int
    extraction_time_seconds: float
    metadata: dict = field(default_factory=dict)


def parse_pdf_file(absolute_file_path: str) -> ParsedDocument:
    extraction_start_time = time.perf_counter()
    pdf_document = fitz.open(absolute_file_path)
    page_text_blocks = [
        pdf_document.load_page(index).get_text("text")
        for index in range(pdf_document.page_count)
    ]
    page_count = pdf_document.page_count
    pdf_document.close()
    return ParsedDocument(
        source_file_path=absolute_file_path,
        file_type="pdf",
        raw_text="\n".join(page_text_blocks),
        page_count=page_count,
        extraction_time_seconds=time.perf_counter() - extraction_start_time,
    )


def parse_docx_file(absolute_file_path: str) -> ParsedDocument:
    extraction_start_time = time.perf_counter()
    docx_document = DocxDocument(absolute_file_path)
    paragraph_texts = [paragraph.text for paragraph in docx_document.paragraphs]
    return ParsedDocument(
        source_file_path=absolute_file_path,
        file_type="docx",
        raw_text="\n".join(paragraph_texts),
        page_count=1,
        extraction_time_seconds=time.perf_counter() - extraction_start_time,
    )


def parse_xlsx_file(absolute_file_path: str) -> ParsedDocument:
    extraction_start_time = time.perf_counter()
    workbook = load_workbook(absolute_file_path, data_only=True)
    all_cell_values: list[str] = []
    for worksheet in workbook.worksheets:
        for row in worksheet.iter_rows(values_only=True):
            row_text = " ".join(str(cell) for cell in row if cell is not None)
            if row_text.strip():
                all_cell_values.append(row_text)
    return ParsedDocument(
        source_file_path=absolute_file_path,
        file_type="xlsx",
        raw_text="\n".join(all_cell_values),
        page_count=len(workbook.worksheets),
        extraction_time_seconds=time.perf_counter() - extraction_start_time,
        metadata={"sheet_names": workbook.sheetnames},
    )


def parse_csv_file(absolute_file_path: str) -> ParsedDocument:
    extraction_start_time = time.perf_counter()
    dataframe = pd.read_csv(absolute_file_path)
    csv_text = dataframe.to_string(index=False)
    return ParsedDocument(
        source_file_path=absolute_file_path,
        file_type="csv",
        raw_text=csv_text,
        page_count=1,
        extraction_time_seconds=time.perf_counter() - extraction_start_time,
        metadata={"row_count": len(dataframe), "column_count": len(dataframe.columns)},
    )


PARSER_DISPATCH_TABLE: dict[str, callable] = {
    ".pdf": parse_pdf_file,
    ".docx": parse_docx_file,
    ".xlsx": parse_xlsx_file,
    ".csv": parse_csv_file,
}


def dispatch_document_parser(absolute_file_path: str) -> ParsedDocument:
    file_extension = pathlib.Path(absolute_file_path).suffix.lower()
    parser_function = PARSER_DISPATCH_TABLE.get(file_extension)
    if parser_function is None:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    return parser_function(absolute_file_path)
```

## 12.5 Logging Runs to MLflow

```python
import json
import time
import mlflow
import tempfile
import pathlib
from dataclasses import asdict

mlflow.set_experiment("/Workspace/Users/team@company.com/nlp-tool/document-extraction")

parsed_document_result = dispatch_document_parser(
    "/Workspace/Users/team@company.com/nlp-tool/data/documents/sample.pdf"
)

quality_score = min(len(parsed_document_result.raw_text) / 1000.0, 1.0)

with mlflow.start_run(run_name="extraction_run") as active_run:
    mlflow.log_param("source_file_path", parsed_document_result.source_file_path)
    mlflow.log_param("file_type", parsed_document_result.file_type)

    mlflow.log_metric("page_count", parsed_document_result.page_count)
    mlflow.log_metric("extraction_time_seconds", parsed_document_result.extraction_time_seconds)
    mlflow.log_metric("quality_score", quality_score)
    mlflow.log_metric("character_count", len(parsed_document_result.raw_text))

    with tempfile.TemporaryDirectory() as temporary_directory:
        artifact_file_path = pathlib.Path(temporary_directory) / "parsed_output.json"
        with open(artifact_file_path, "w", encoding="utf-8") as artifact_file_handle:
            json.dump(asdict(parsed_document_result), artifact_file_handle, indent=2)
        mlflow.log_artifact(str(artifact_file_path), artifact_path="parsed_outputs")

    print(f"Run ID: {active_run.info.run_id}")
```

## 12.6 Setting and Using a Shared Experiment

```python
import mlflow

SHARED_EXPERIMENT_PATH = "/Workspace/Users/team@company.com/nlp-tool/document-extraction"

mlflow.set_experiment(SHARED_EXPERIMENT_PATH)

experiment_definition = mlflow.get_experiment_by_name(SHARED_EXPERIMENT_PATH)
print(f"Experiment ID: {experiment_definition.experiment_id}")
print(f"Artifact location: {experiment_definition.artifact_location}")

with mlflow.start_run(run_name="shared_experiment_validation_run") as active_run:
    mlflow.log_param("pipeline_version", "1.0.0")
    mlflow.log_param("runtime_version", "14.3.x-scala2.12")
    mlflow.log_param("document_schema_version", "v2")

    mlflow.log_metric("documents_processed", 42)
    mlflow.log_metric("average_extraction_time_seconds", 0.38)
    mlflow.log_metric("average_quality_score", 0.91)
    mlflow.log_metric("pipeline_success_rate", 0.98)

    mlflow.set_tag("pipeline_stage", "extraction")
    mlflow.set_tag("triggered_by", "scheduled_job")

    print(f"Active run ID: {active_run.info.run_id}")
```

## 12.7 Evaluating an Extraction Workflow

```python
import mlflow
from dataclasses import dataclass


@dataclass
class ExtractionEvaluationResult:
    document_identifier: str
    completeness_score: float
    accuracy_score: float
    missing_term_count: int


def compute_completeness_score(extracted_text: str, ground_truth_text: str) -> float:
    ground_truth_tokens = set(ground_truth_text.lower().split())
    extracted_tokens = set(extracted_text.lower().split())
    if not ground_truth_tokens:
        return 0.0
    return len(ground_truth_tokens & extracted_tokens) / len(ground_truth_tokens)


def compute_accuracy_score(extracted_text: str, ground_truth_text: str) -> float:
    ground_truth_tokens = set(ground_truth_text.lower().split())
    extracted_tokens = set(extracted_text.lower().split())
    union_token_count = len(ground_truth_tokens | extracted_tokens)
    if union_token_count == 0:
        return 0.0
    return len(ground_truth_tokens & extracted_tokens) / union_token_count


ground_truth_corpus = [
    {"document_identifier": "doc_001", "ground_truth_text": "invoice total amount due payment"},
    {"document_identifier": "doc_002", "ground_truth_text": "contract agreement terms conditions parties"},
]

extraction_corpus = [
    {"document_identifier": "doc_001", "extracted_text": "invoice total amount due payment date"},
    {"document_identifier": "doc_002", "extracted_text": "contract agreement terms parties signature"},
]

mlflow.set_experiment("/Workspace/Users/team@company.com/nlp-tool/document-extraction")

with mlflow.start_run(run_name="extraction_evaluation_run") as active_run:
    evaluation_results = []

    for ground_truth_entry, extraction_entry in zip(ground_truth_corpus, extraction_corpus):
        completeness = compute_completeness_score(
            extraction_entry["extracted_text"],
            ground_truth_entry["ground_truth_text"],
        )
        accuracy = compute_accuracy_score(
            extraction_entry["extracted_text"],
            ground_truth_entry["ground_truth_text"],
        )
        ground_truth_token_set = set(ground_truth_entry["ground_truth_text"].lower().split())
        extracted_token_set = set(extraction_entry["extracted_text"].lower().split())
        missing_term_count = len(ground_truth_token_set - extracted_token_set)

        evaluation_results.append(
            ExtractionEvaluationResult(
                document_identifier=ground_truth_entry["document_identifier"],
                completeness_score=completeness,
                accuracy_score=accuracy,
                missing_term_count=missing_term_count,
            )
        )

    average_completeness = sum(r.completeness_score for r in evaluation_results) / len(evaluation_results)
    average_accuracy = sum(r.accuracy_score for r in evaluation_results) / len(evaluation_results)
    total_missing_terms = sum(r.missing_term_count for r in evaluation_results)

    mlflow.log_metric("average_completeness_score", average_completeness)
    mlflow.log_metric("average_accuracy_score", average_accuracy)
    mlflow.log_metric("total_missing_term_count", total_missing_terms)
    mlflow.log_metric("documents_evaluated", len(evaluation_results))
```

## 12.8 Chunking Extracted Text

```python
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    chunk_index: int
    chunk_text: str
    start_character_offset: int
    end_character_offset: int
    source_document_path: str
    chunk_metadata: dict = field(default_factory=dict)


def chunk_document_text(
    raw_text: str,
    source_document_path: str,
    chunk_size: int = 512,
    overlap_size: int = 64,
    additional_metadata: dict | None = None,
) -> list[TextChunk]:
    if overlap_size >= chunk_size:
        raise ValueError("overlap_size must be strictly less than chunk_size")

    resolved_metadata = additional_metadata or {}
    stride = chunk_size - overlap_size
    text_length = len(raw_text)
    text_chunks: list[TextChunk] = []
    chunk_index = 0
    current_position = 0

    while current_position < text_length:
        end_position = min(current_position + chunk_size, text_length)
        chunk_text = raw_text[current_position:end_position]

        text_chunks.append(
            TextChunk(
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                start_character_offset=current_position,
                end_character_offset=end_position,
                source_document_path=source_document_path,
                chunk_metadata={
                    **resolved_metadata,
                    "total_text_length": text_length,
                    "chunk_size_setting": chunk_size,
                    "overlap_size_setting": overlap_size,
                },
            )
        )

        chunk_index += 1
        current_position += stride

    return text_chunks
```

## 12.9 Simple Retrieval with Databricks Vector Search

```python
from databricks.vector_search.client import VectorSearchClient
from sentence_transformers import SentenceTransformer


VECTOR_SEARCH_ENDPOINT_NAME = "nlp-tool-vector-search-endpoint"
VECTOR_SEARCH_INDEX_NAME = "main.nlp_tool.document_chunks_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_RESULTS = 5

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
vector_search_client = VectorSearchClient()


def embed_query_text(query_text: str) -> list[float]:
    embedding_array = embedding_model.encode(query_text, normalize_embeddings=True)
    return embedding_array.tolist()


def retrieve_similar_chunks(query_text: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    query_embedding_vector = embed_query_text(query_text)
    vector_search_index = vector_search_client.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=VECTOR_SEARCH_INDEX_NAME,
    )
    search_response = vector_search_index.similarity_search(
        query_vector=query_embedding_vector,
        columns=["chunk_index", "chunk_text", "source_document_path", "start_character_offset"],
        num_results=top_k,
    )
    return search_response.get("result", {}).get("data_array", [])


def format_retrieval_results(raw_result_rows: list[dict]) -> list[dict]:
    return [
        {
            "rank": rank_position + 1,
            "chunk_text": result_row[1],
            "source_document_path": result_row[2],
            "similarity_score": result_row[-1],
        }
        for rank_position, result_row in enumerate(raw_result_rows)
    ]


user_query_text = "What is the total invoice amount due?"
raw_retrieval_results = retrieve_similar_chunks(user_query_text)
formatted_retrieval_results = format_retrieval_results(raw_retrieval_results)

for retrieval_result in formatted_retrieval_results:
    print(f"Rank {retrieval_result['rank']} | Score: {retrieval_result['similarity_score']:.4f}")
    print(f"Source: {retrieval_result['source_document_path']}")
    print(f"Text: {retrieval_result['chunk_text'][:120]}")
    print()
```

## 12.10 Library Dependencies

```
PyMuPDF>=1.23.0
python-docx>=1.1.0
openpyxl>=3.1.2
pandas>=2.1.0
mlflow>=2.10.0
sentence-transformers>=2.6.0
databricks-vectorsearch>=0.22
tiktoken>=0.7.0
pdfplumber>=0.11.0
python-magic>=0.4.27
chardet>=5.2.0
```

---

# Section 13: Final Recommendation

## Recommended Target Architecture

**Hybrid two-tier system:**

- **Tier 1 (Custom Pipeline):** Python-based document parsing (PyMuPDF, pdfplumber, python-docx, pandas) producing normalized `ParsedDocument` records, chunked and embedded into Vector Search for RAG-based Q&A via a Mosaic AI Agent. This handles the "Textract-like" requirement.

- **Tier 2 (Genie):** Natural-language SQL over structured extraction tables in Unity Catalog. Activated in Phase 4 once extractions land in well-modeled, commented Delta tables.

- **Connective tissue:** Mosaic AI Agent Framework with three tools (semantic retrieval, structured lookup, metadata lookup), MLflow experiments for evaluation/tracing, and Databricks Workflows for orchestration.

## Recommended Phase 1 Build

Start with the simplest thing that proves the core value proposition:

1. **Workspace folder structure** with notebooks, src, sample_data, configs, eval
2. **Python parsing pipeline** on the driver node -- PDF, DOCX, XLSX, CSV, TXT
3. **Normalized schema** (ParsedDocument dataclass) and Delta table writes
4. **Character/token-aware chunking** with 512-token windows and 64-token overlap
5. **Brute-force retrieval** (cosine similarity over embeddings) -- no Vector Search yet
6. **Workspace MLflow experiment** with baseline extraction and retrieval metrics
7. **10-20 test documents** across 3+ file types

Phase 1 should take 2-3 weeks and produce a working end-to-end demo in a single notebook session.

## Databricks Services to Use Immediately

| Service | Phase 1 Role |
|---|---|
| **Databricks Notebooks** | Primary development environment |
| **Workspace Files** | Store sample documents and .py modules |
| **Delta Lake** | Persist parsed documents and chunks |
| **MLflow Experiments** | Track extraction quality and retrieval baselines |
| **Foundation Model API** | Generate embeddings (databricks-bge-large-en) |
| **%pip install** | Install parsing libraries on cluster |

## Services to Defer Until Files Are in Better Storage

| Service | Defer Until | Why |
|---|---|---|
| **Databricks Document Parsing** | Phase 2 (files in UC Volumes) | Requires Unity Catalog-enabled inputs |
| **Information Extraction** | Phase 2 (files in UC Volumes) | Requires UC + serverless/public-preview |
| **Vector Search** | Phase 2 (chunks in UC Delta table) | Requires UC-managed source table |
| **Genie** | Phase 4 (structured extractions in UC tables with comments) | Requires well-modeled UC tables with Pro/Serverless SQL Warehouse |
| **Model Serving (Agent)** | Phase 3 (agent built and evaluated) | Requires working agent with passing eval scores |
| **Inference Tables** | Phase 3 (agent deployed) | Auto-created by Model Serving |
| **Databricks Workflows** | Phase 4 (production orchestration) | Not needed for notebook-driven dev |

## Key Technical Risks

1. **PDF parsing quality variance**: Scanned PDFs, complex layouts, and embedded images will reduce extraction quality. Mitigate with OCR fallback and Databricks Document Parsing in Phase 2.
2. **Serverless executor limitations**: Cannot read workspace files. All parsing must stay on the driver. Production must use UC Volumes.
3. **Information Extraction instability**: Public preview. Pin versions. Build custom extraction first; layer native features as enhancement.
4. **Agent hallucination**: Highest business risk. Non-negotiable controls: grounding mandate, citation requirement, tool-call enforcement, faithfulness deployment gate.
5. **Cost at scale**: Track token usage per document from day 1. Set budget alerts early.

---

*This plan is implementation-ready and designed to be executed as-is by a Databricks engineering team. Each section can be handed directly to the responsible engineer or workstream.*
