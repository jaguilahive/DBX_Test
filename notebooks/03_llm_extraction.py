# Databricks notebook source

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import os
import json
import time
from pipeline import process_file
from llm import call_llm_json, build_extraction_prompt
from postprocess import normalize_fields, validate_extraction, clean_extracted_text

# COMMAND ----------

INVOICE_SCHEMA = {
    "invoice_number": "string",
    "vendor_name": "string",
    "invoice_date": "date",
    "due_date": "date",
    "total_amount": "currency",
    "line_items": "list of items with description and amount",
}

INVOICE_FIELD_TYPES = {
    "invoice_number": "text",
    "vendor_name": "text",
    "invoice_date": "date",
    "due_date": "date",
    "total_amount": "currency",
}

REQUIRED_INVOICE_FIELDS = ["invoice_number", "vendor_name", "total_amount"]

# COMMAND ----------

sample_files = []
for subdir in ["pdfs", "docx", "xlsx"]:
    subdir_path = os.path.join(SAMPLE_DATA_DIR, subdir)
    if os.path.isdir(subdir_path):
        for filename in sorted(os.listdir(subdir_path)):
            filepath = os.path.join(subdir_path, filename)
            if os.path.isfile(filepath) and not filename.startswith("."):
                sample_files.append(filepath)

print(f"Files to process: {len(sample_files)}")

# COMMAND ----------

extraction_results = []

for file_path in sample_files:
    print(f"\nProcessing: {os.path.basename(file_path)}")

    doc, chunks = process_file(file_path)
    if not doc.is_successful:
        print(f"  SKIPPED: extraction failed")
        continue

    cleaned_text = clean_extracted_text(doc.extracted_text)
    truncated_text = cleaned_text[:8000]

    prompt = build_extraction_prompt(truncated_text, INVOICE_SCHEMA)

    system_prompt = (
        "You are a document extraction system. "
        "Return only valid JSON. Do not include any text outside the JSON object."
    )

    try:
        start = time.perf_counter()
        raw_fields = call_llm_json(
            prompt,
            system_prompt=system_prompt,
            endpoint_name=MODEL_CONFIG["llm"]["endpoint_name"],
            max_tokens=MODEL_CONFIG["llm"]["max_tokens"],
        )
        extraction_time = time.perf_counter() - start

        normalized = normalize_fields(raw_fields, INVOICE_FIELD_TYPES)
        validated, validation_errors = validate_extraction(normalized, REQUIRED_INVOICE_FIELDS)

        extraction_results.append({
            "file_name": doc.file_name,
            "doc_id": doc.doc_id,
            "extracted_fields": validated,
            "validation_errors": validation_errors,
            "extraction_time_seconds": round(extraction_time, 2),
        })

        print(f"  Extracted: {json.dumps(validated, indent=2)}")
        if validation_errors:
            print(f"  Validation errors: {validation_errors}")

    except Exception as exc:
        print(f"  ERROR: {str(exc)}")
        extraction_results.append({
            "file_name": doc.file_name,
            "doc_id": doc.doc_id,
            "extracted_fields": {},
            "validation_errors": [{"error": str(exc)}],
            "extraction_time_seconds": 0,
        })

# COMMAND ----------

import pandas as pd

results_flat = []
for result in extraction_results:
    flat = {
        "file_name": result["file_name"],
        "extraction_time_seconds": result["extraction_time_seconds"],
        "error_count": len(result["validation_errors"]),
    }
    flat.update(result["extracted_fields"])
    results_flat.append(flat)

results_df = pd.DataFrame(results_flat)
display(results_df)

# COMMAND ----------

import mlflow

with mlflow.start_run(run_name="llm_extraction"):
    mlflow.log_param("llm_endpoint", MODEL_CONFIG["llm"]["endpoint_name"])
    mlflow.log_param("schema_type", "invoice")
    mlflow.log_metric("documents_processed", len(extraction_results))
    mlflow.log_metric("avg_extraction_time", sum(r["extraction_time_seconds"] for r in extraction_results) / max(len(extraction_results), 1))
    mlflow.log_metric("total_validation_errors", sum(len(r["validation_errors"]) for r in extraction_results))
    mlflow.set_tag("pipeline_stage", "llm_extraction")
    print("Extraction metrics logged to MLflow.")
