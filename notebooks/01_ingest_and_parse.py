# Databricks notebook source

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import os
from parsing import detect_file_type, compute_quality_score, log_extraction_result
from pipeline import process_file, process_directory, documents_to_records, chunks_to_records
from schemas import ParsedDocument

# COMMAND ----------

sample_subdirs = ["pdfs", "docx", "xlsx", "txt"]
all_sample_files = []

for subdir in sample_subdirs:
    subdir_path = os.path.join(SAMPLE_DATA_DIR, subdir)
    if os.path.isdir(subdir_path):
        for filename in sorted(os.listdir(subdir_path)):
            filepath = os.path.join(subdir_path, filename)
            if os.path.isfile(filepath) and not filename.startswith("."):
                all_sample_files.append(filepath)

print(f"Found {len(all_sample_files)} sample files:")
for fp in all_sample_files:
    mime, parser = detect_file_type(fp)
    print(f"  {os.path.basename(fp)} -> {parser} ({mime})")

# COMMAND ----------

all_docs = []
all_chunks = []

for file_path in all_sample_files:
    print(f"Processing: {os.path.basename(file_path)}")
    doc, chunks = process_file(
        file_path,
        chunk_size=CHUNKING_CONFIG["chunk_size"],
        chunk_overlap=CHUNKING_CONFIG["chunk_overlap"],
    )
    all_docs.append(doc)
    all_chunks.extend(chunks)
    print(f"  -> {doc.page_count} pages, {doc.table_count} tables, {len(chunks)} chunks, quality={doc.extraction_quality_score:.3f}")

# COMMAND ----------

print(f"\nTotal documents: {len(all_docs)}")
print(f"Total chunks: {len(all_chunks)}")
print(f"Successful: {sum(1 for d in all_docs if d.is_successful)}")
print(f"Failed: {sum(1 for d in all_docs if not d.is_successful)}")

# COMMAND ----------

import pandas as pd

doc_records = documents_to_records(all_docs)
doc_df = pd.DataFrame(doc_records)
display(doc_df)

# COMMAND ----------

chunk_records = chunks_to_records(all_chunks)
chunk_df = pd.DataFrame(chunk_records)
print(f"Chunk stats: min_tokens={chunk_df['token_count'].min()}, max_tokens={chunk_df['token_count'].max()}, avg_tokens={chunk_df['token_count'].mean():.0f}")
display(chunk_df.head(20))

# COMMAND ----------

import mlflow

with mlflow.start_run(run_name="ingest_baseline"):
    mlflow.log_metric("total_documents", len(all_docs))
    mlflow.log_metric("total_chunks", len(all_chunks))
    mlflow.log_metric("success_count", sum(1 for d in all_docs if d.is_successful))
    mlflow.log_metric("failure_count", sum(1 for d in all_docs if not d.is_successful))
    mlflow.log_metric("avg_quality_score", sum(d.extraction_quality_score for d in all_docs) / max(len(all_docs), 1))
    mlflow.log_metric("avg_chunks_per_doc", len(all_chunks) / max(len(all_docs), 1))

    for doc in all_docs:
        mlflow.log_metric(f"quality_{doc.file_name}", doc.extraction_quality_score)

    mlflow.set_tag("pipeline_stage", "ingestion")
    mlflow.set_tag("document_count", str(len(all_docs)))
    print("Metrics logged to MLflow.")
