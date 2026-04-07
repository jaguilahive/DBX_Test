import sys
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from schemas import generate_doc_id
from parsing import (
    detect_file_type, parse_pdf, parse_docx, parse_doc_legacy,
    parse_spreadsheet, parse_text, compute_quality_score,
    log_extraction_result, _failed_document,
)
from chunking import DocumentChunker

logger = logging.getLogger("extraction_pipeline")

PARSER_MAP = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "doc_legacy": parse_doc_legacy,
    "csv": lambda fp, did: parse_spreadsheet(fp, did, "csv"),
    "xlsx": lambda fp, did: parse_spreadsheet(fp, did, "xlsx"),
    "xls": lambda fp, did: parse_spreadsheet(fp, did, "xls"),
    "text": parse_text,
}


def process_file(file_path, chunk_size=512, chunk_overlap=64):
    doc_id = generate_doc_id()

    mime_type, parser_name = detect_file_type(file_path)

    if parser_name == "unsupported":
        doc = _failed_document(
            file_path, doc_id, mime_type,
            Path(file_path).suffix.lower(),
            ValueError(f"Unsupported file type: {mime_type}"), 0,
        )
        log_extraction_result(doc)
        return doc, []

    parser_fn = PARSER_MAP[parser_name]
    doc = parser_fn(file_path, doc_id)

    doc.extraction_quality_score = compute_quality_score(doc)

    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = chunker.chunk_document(doc)

    log_extraction_result(doc)

    return doc, chunks


def process_directory(directory_path, chunk_size=512, chunk_overlap=64):
    all_docs = []
    all_chunks = []

    for entry in sorted(os.listdir(directory_path)):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path) and not entry.startswith("."):
            try:
                doc, chunks = process_file(full_path, chunk_size, chunk_overlap)
                all_docs.append(doc)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.error(f"PIPELINE_ERROR file={entry} error={str(exc)}")

    return all_docs, all_chunks


def process_directory_parallel(directory_path, max_workers=4, chunk_size=512, chunk_overlap=64):
    files = [
        os.path.join(directory_path, f)
        for f in sorted(os.listdir(directory_path))
        if os.path.isfile(os.path.join(directory_path, f))
        and not f.startswith(".")
    ]

    all_docs = []
    all_chunks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_file, fp, chunk_size, chunk_overlap): fp
            for fp in files
        }
        for future in as_completed(futures):
            try:
                doc, chunks = future.result(timeout=300)
                all_docs.append(doc)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.error(f"PARALLEL_ERROR file={futures[future]} error={str(exc)}")

    return all_docs, all_chunks


def documents_to_records(docs):
    records = []
    for doc in docs:
        record = {
            "doc_id": doc.doc_id,
            "source_path": doc.source_path,
            "file_name": doc.file_name,
            "file_type": doc.file_type,
            "file_extension": doc.file_extension,
            "file_size_bytes": doc.file_size_bytes,
            "extracted_text": doc.extracted_text,
            "page_count": doc.page_count,
            "table_count": doc.table_count,
            "extraction_timestamp": doc.extraction_timestamp,
            "extraction_duration_ms": doc.extraction_duration_ms,
            "extraction_method": doc.extraction_method,
            "extraction_quality_score": doc.extraction_quality_score,
            "ocr_applied": doc.ocr_applied,
            "is_successful": doc.is_successful,
            "error_count": len(doc.errors),
        }
        records.append(record)
    return records


def chunks_to_records(chunks):
    records = []
    for chunk in chunks:
        record = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "text": chunk.text,
            "token_count": chunk.token_count,
            "char_count": chunk.char_count,
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "source_path": chunk.source_path,
            "file_name": chunk.file_name,
            "file_type": chunk.file_type,
            "contains_table": chunk.contains_table,
            "extraction_timestamp": chunk.extraction_timestamp,
        }
        records.append(record)
    return records


def write_to_delta(docs, chunks, spark, docs_table, chunks_table):
    import pandas as pd

    if docs:
        doc_records = documents_to_records(docs)
        doc_df = spark.createDataFrame(pd.DataFrame(doc_records))
        doc_df.write.format("delta").mode("append").saveAsTable(docs_table)

    if chunks:
        chunk_records = chunks_to_records(chunks)
        chunk_df = spark.createDataFrame(pd.DataFrame(chunk_records))
        chunk_df.write.format("delta").mode("append").saveAsTable(chunks_table)
