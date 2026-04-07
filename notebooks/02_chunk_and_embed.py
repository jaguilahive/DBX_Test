# Databricks notebook source

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

from pipeline import process_directory
from embeddings import generate_embeddings_batch, embed_chunks
import os

# COMMAND ----------

all_docs = []
all_chunks = []

for subdir in ["pdfs", "docx", "xlsx", "txt"]:
    subdir_path = os.path.join(SAMPLE_DATA_DIR, subdir)
    if os.path.isdir(subdir_path):
        docs, chunks = process_directory(
            subdir_path,
            chunk_size=CHUNKING_CONFIG["chunk_size"],
            chunk_overlap=CHUNKING_CONFIG["chunk_overlap"],
        )
        all_docs.extend(docs)
        all_chunks.extend(chunks)

print(f"Parsed {len(all_docs)} documents into {len(all_chunks)} chunks")

# COMMAND ----------

embedding_endpoint = MODEL_CONFIG["embedding"]["endpoint_name"]
embedding_batch_size = MODEL_CONFIG["embedding"]["batch_size"]

chunk_texts = [chunk.text for chunk in all_chunks]

print(f"Generating embeddings for {len(chunk_texts)} chunks using {embedding_endpoint}...")
embeddings = generate_embeddings_batch(
    chunk_texts,
    endpoint_name=embedding_endpoint,
    batch_size=embedding_batch_size,
)
print(f"Generated {len(embeddings)} embeddings, dimension={len(embeddings[0]) if embeddings else 0}")

# COMMAND ----------

import pandas as pd
from pipeline import chunks_to_records

chunk_records = chunks_to_records(all_chunks)
for idx, record in enumerate(chunk_records):
    record["embedding"] = embeddings[idx]

embedding_df = pd.DataFrame(chunk_records)
print(f"Embedding DataFrame shape: {embedding_df.shape}")
display(embedding_df[["chunk_id", "doc_id", "file_name", "chunk_index", "token_count"]].head(20))

# COMMAND ----------

import mlflow

with mlflow.start_run(run_name="chunk_and_embed"):
    mlflow.log_param("embedding_endpoint", embedding_endpoint)
    mlflow.log_param("chunk_size", CHUNKING_CONFIG["chunk_size"])
    mlflow.log_param("chunk_overlap", CHUNKING_CONFIG["chunk_overlap"])
    mlflow.log_metric("total_chunks", len(all_chunks))
    mlflow.log_metric("total_embeddings", len(embeddings))
    mlflow.log_metric("embedding_dimension", len(embeddings[0]) if embeddings else 0)
    mlflow.set_tag("pipeline_stage", "chunking_embedding")
    print("Embedding metrics logged to MLflow.")
