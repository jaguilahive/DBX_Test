# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install PyMuPDF==1.24.3 pdfplumber==0.11.0 python-docx==1.1.0 openpyxl>=3.1.2 tiktoken==0.7.0 chardet>=5.2.0 pyyaml>=6.0

# COMMAND ----------

import sys
import os
import glob

print(f"DEBUG: os.getcwd() = {os.getcwd()}")
print(f"DEBUG: __name__ = {__name__}")

# COMMAND ----------

def _find_src_dir():
    cwd = os.getcwd()

    candidate_a = os.path.join(cwd, "..", "src")
    if os.path.isfile(os.path.join(candidate_a, "schemas.py")):
        return os.path.abspath(candidate_a), os.path.abspath(os.path.join(candidate_a, ".."))

    candidate_b = os.path.join(cwd, "src")
    if os.path.isfile(os.path.join(candidate_b, "schemas.py")):
        return os.path.abspath(candidate_b), os.path.abspath(cwd)

    workspace_patterns = [
        "/Workspace/Repos/*/DBX_Test/src",
        "/Workspace/Repos/*/*/src",
        "/Workspace/Users/*/DBX_Test/src",
        "/Workspace/Users/*/*/src",
        "/Workspace/Shared/*/src",
    ]
    for pattern in workspace_patterns:
        matches = glob.glob(pattern)
        for match in matches:
            if os.path.isfile(os.path.join(match, "schemas.py")):
                return os.path.abspath(match), os.path.abspath(os.path.join(match, ".."))

    raise RuntimeError(
        f"Cannot find src/schemas.py. "
        f"os.getcwd()={cwd}. "
        f"Checked: {cwd}/../src, {cwd}/src, and Workspace glob patterns. "
        f"Make sure the full project (src/, notebooks/, configs/, etc.) "
        f"is uploaded to your Databricks workspace."
    )


SRC_DIR, PROJECT_ROOT = _find_src_dir()

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print(f"DEBUG: PROJECT_ROOT = {PROJECT_ROOT}")
print(f"DEBUG: SRC_DIR = {SRC_DIR}")
print(f"DEBUG: schemas.py exists = {os.path.isfile(os.path.join(SRC_DIR, 'schemas.py'))}")
print(f"DEBUG: parsing.py exists = {os.path.isfile(os.path.join(SRC_DIR, 'parsing.py'))}")

# COMMAND ----------

from schemas import ParsedDocument, DocumentChunk
from parsing import detect_file_type
print(f"DEBUG: import test passed - detect_file_type = {detect_file_type}")

# COMMAND ----------

SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "sample_data")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")

# COMMAND ----------

import yaml

def load_config(filename):
    config_path = os.path.join(CONFIG_DIR, filename)
    if not os.path.isfile(config_path):
        print(f"WARNING: config file not found: {config_path}")
        return {}
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


MODEL_CONFIG = load_config("model_config.yaml")
CHUNKING_CONFIG = load_config("chunking_config.yaml")
EVAL_CONFIG = load_config("eval_config.yaml")

# COMMAND ----------

print(f"PROJECT_ROOT:    {PROJECT_ROOT}")
print(f"SRC_DIR:         {SRC_DIR}")
print(f"SAMPLE_DATA_DIR: {SAMPLE_DATA_DIR}")
print(f"CONFIG_DIR:      {CONFIG_DIR}")
print(f"EVAL_DIR:        {EVAL_DIR}")

sample_files_found = 0
for subdir in ["pdfs", "docx", "xlsx", "txt"]:
    subdir_path = os.path.join(SAMPLE_DATA_DIR, subdir)
    if os.path.isdir(subdir_path):
        files = [f for f in os.listdir(subdir_path) if not f.startswith(".")]
        sample_files_found += len(files)
        print(f"  sample_data/{subdir}/: {len(files)} files")

print(f"Total sample files: {sample_files_found}")
print(f"Setup complete.")
