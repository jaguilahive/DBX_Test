# Databricks notebook source

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import os
import json
from pipeline import process_directory
from eval_utils import (
    evaluate_extraction_batch, compute_aggregate_metrics,
    log_evaluation_to_mlflow, compute_completeness_score, compute_accuracy_score,
)

# COMMAND ----------

ground_truth_dir = os.path.join(EVAL_DIR, "ground_truth")
print(f"Ground truth directory: {ground_truth_dir}")

gt_files = [f for f in os.listdir(ground_truth_dir) if f.endswith(".json")] if os.path.isdir(ground_truth_dir) else []
print(f"Ground truth files: {len(gt_files)}")

for gt_file in gt_files:
    with open(os.path.join(ground_truth_dir, gt_file), "r") as fh:
        gt = json.load(fh)
    print(f"  {gt_file}: {list(gt.keys())}")

# COMMAND ----------

all_docs = []
all_chunks = []

for subdir in ["pdfs", "docx", "xlsx", "txt"]:
    subdir_path = os.path.join(SAMPLE_DATA_DIR, subdir)
    if os.path.isdir(subdir_path):
        docs, chunks = process_directory(subdir_path)
        all_docs.extend(docs)
        all_chunks.extend(chunks)

print(f"Parsed {len(all_docs)} documents")

# COMMAND ----------

evaluation_results = evaluate_extraction_batch(all_docs, ground_truth_dir)

print(f"\nEvaluation results for {len(evaluation_results)} documents:")
for result in evaluation_results:
    print(f"  {result['file_name']}: completeness={result['completeness_score']:.3f}, accuracy={result['accuracy_score']:.3f}, quality={result['extraction_quality_score']:.3f}")

# COMMAND ----------

aggregate = compute_aggregate_metrics(evaluation_results)

print("\nAggregate Metrics:")
for metric_name, metric_value in aggregate.items():
    print(f"  {metric_name}: {metric_value}")

# COMMAND ----------

thresholds = EVAL_CONFIG["thresholds"]

print("\nThreshold Check:")
checks = {
    "extraction_completeness": aggregate.get("average_completeness", 0) >= thresholds["extraction_completeness"],
    "failure_rate": aggregate.get("failure_rate", 1) <= thresholds["failure_rate"],
}

for check_name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {check_name}: {status}")

# COMMAND ----------

experiment_path = EVAL_CONFIG["experiment_paths"]["extraction_eval"]
run_id = log_evaluation_to_mlflow(evaluation_results, aggregate, experiment_path)
print(f"Evaluation logged to MLflow run: {run_id}")
