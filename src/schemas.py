import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dataclasses import dataclass, field
from typing import Optional
import uuid
import hashlib


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


def generate_doc_id():
    return str(uuid.uuid4())


def generate_table_id():
    return str(uuid.uuid4())


def generate_chunk_id(doc_id, chunk_index):
    raw = f"{doc_id}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def table_to_markdown(table):
    lines = []
    if table.sheet_name:
        lines.append(f"**Table from sheet: {table.sheet_name}**\n")
    lines.append("| " + " | ".join(table.headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(table.headers)) + " |")
    for row in table.rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def compute_table_confidence(headers, rows):
    if not headers or not rows:
        return 0.0
    total_cells = len(rows) * len(headers)
    non_empty = sum(1 for row in rows for cell in row if cell.strip())
    fill_rate = non_empty / max(total_cells, 1)
    expected_cols = len(headers)
    consistent_rows = sum(1 for row in rows if len(row) == expected_cols)
    consistency = consistent_rows / max(len(rows), 1)
    return round(min(fill_rate * 0.6 + consistency * 0.4, 1.0), 3)
