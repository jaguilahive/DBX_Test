# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install PyMuPDF==1.24.3 pdfplumber==0.11.0 python-docx==1.1.0 openpyxl>=3.1.2 tiktoken==0.7.0 chardet>=5.2.0 pyyaml>=6.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import os
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "sample_data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")

# COMMAND ----------

%load_ext autoreload
%autoreload 2

# COMMAND ----------

def load_config(filename):
    config_path = os.path.join(CONFIG_DIR, filename)
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


MODEL_CONFIG = load_config("model_config.yaml")
CHUNKING_CONFIG = load_config("chunking_config.yaml")
EVAL_CONFIG = load_config("eval_config.yaml")

# COMMAND ----------

MLFLOW_EXPERIMENT_PATH = "/Workspace/Users/team@company.com/doc-understanding/experiments/extraction-eval"

import mlflow
mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

# COMMAND ----------

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"SRC_DIR: {SRC_DIR}")
print(f"SAMPLE_DATA_DIR: {SAMPLE_DATA_DIR}")
print(f"CONFIG_DIR: {CONFIG_DIR}")
print(f"Setup complete.")
