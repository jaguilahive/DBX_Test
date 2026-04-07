# Databricks notebook source

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import os
import time
from pipeline import process_file
from chunking import DocumentChunker
from llm import call_llm, build_qa_prompt
from postprocess import clean_extracted_text

# COMMAND ----------

print("=== DATABRICKS NLP DOCUMENT UNDERSTANDING - END TO END DEMO ===\n")

sample_files = []
for subdir in ["pdfs", "docx", "xlsx", "txt"]:
    subdir_path = os.path.join(SAMPLE_DATA_DIR, subdir)
    if os.path.isdir(subdir_path):
        for fname in sorted(os.listdir(subdir_path)):
            fpath = os.path.join(subdir_path, fname)
            if os.path.isfile(fpath) and not fname.startswith("."):
                sample_files.append(fpath)

print(f"Sample files available: {len(sample_files)}")
for fp in sample_files:
    print(f"  {os.path.basename(fp)}")

# COMMAND ----------

print("\n=== STEP 1: INGEST AND PARSE ===\n")

all_docs = []
all_chunks = []
total_start = time.perf_counter()

for file_path in sample_files:
    doc, chunks = process_file(
        file_path,
        chunk_size=CHUNKING_CONFIG["chunk_size"],
        chunk_overlap=CHUNKING_CONFIG["chunk_overlap"],
    )
    all_docs.append(doc)
    all_chunks.extend(chunks)

    status = "OK" if doc.is_successful else "FAILED"
    print(f"  [{status}] {doc.file_name}: {doc.page_count} pages, {doc.table_count} tables, {len(chunks)} chunks, quality={doc.extraction_quality_score:.3f}")

total_parse_time = time.perf_counter() - total_start
print(f"\nTotal parse time: {total_parse_time:.2f}s")
print(f"Documents: {len(all_docs)}, Chunks: {len(all_chunks)}")

# COMMAND ----------

print("\n=== STEP 2: CHUNK SUMMARY ===\n")

import pandas as pd
from pipeline import chunks_to_records

chunk_records = chunks_to_records(all_chunks)
chunk_df = pd.DataFrame(chunk_records)

print(f"Total chunks: {len(chunk_df)}")
print(f"Average tokens per chunk: {chunk_df['token_count'].mean():.0f}")
print(f"Table chunks: {chunk_df['contains_table'].sum()}")
print(f"Chunks by file type:")
display(chunk_df.groupby("file_type").agg(
    chunk_count=("chunk_id", "count"),
    avg_tokens=("token_count", "mean"),
).reset_index())

# COMMAND ----------

print("\n=== STEP 3: DOCUMENT Q&A (requires Databricks Model Serving) ===\n")

demo_questions = [
    "What is the total amount on the invoice?",
    "Who are the parties in the contract?",
    "Summarize the key findings in the report.",
]

for question in demo_questions:
    print(f"\nQ: {question}")

    relevant_chunks = all_chunks[:5]

    try:
        prompt = build_qa_prompt(question, relevant_chunks)
        answer = call_llm(
            prompt,
            system_prompt="Answer based ONLY on the provided context. Cite sources.",
            endpoint_name=MODEL_CONFIG["llm"]["endpoint_name"],
        )
        print(f"A: {answer}\n")
    except Exception as exc:
        print(f"A: [LLM not available in local testing: {type(exc).__name__}]\n")

# COMMAND ----------

print("\n=== STEP 4: LOG TO MLFLOW ===\n")

import mlflow

with mlflow.start_run(run_name="end_to_end_demo"):
    mlflow.log_param("chunk_size", CHUNKING_CONFIG["chunk_size"])
    mlflow.log_param("chunk_overlap", CHUNKING_CONFIG["chunk_overlap"])
    mlflow.log_param("llm_endpoint", MODEL_CONFIG["llm"]["endpoint_name"])

    mlflow.log_metric("total_documents", len(all_docs))
    mlflow.log_metric("total_chunks", len(all_chunks))
    mlflow.log_metric("total_parse_time_seconds", round(total_parse_time, 2))
    mlflow.log_metric("success_rate", sum(1 for d in all_docs if d.is_successful) / max(len(all_docs), 1))
    mlflow.log_metric("avg_quality", sum(d.extraction_quality_score for d in all_docs) / max(len(all_docs), 1))

    mlflow.set_tag("demo", "end_to_end")
    print("Demo metrics logged to MLflow.")

# COMMAND ----------

print("\n=== DEMO COMPLETE ===")
print(f"Documents processed: {len(all_docs)}")
print(f"Chunks generated: {len(all_chunks)}")
print(f"Total time: {total_parse_time:.2f}s")
