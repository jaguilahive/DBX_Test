# Databricks notebook source

# COMMAND ----------

# MAGIC %run ./00_setup

# COMMAND ----------

import json
import os
from postprocess import normalize_fields, validate_extraction, clean_extracted_text

# COMMAND ----------

FIELD_TYPES = {
    "invoice_number": "text",
    "vendor_name": "text",
    "invoice_date": "date",
    "due_date": "date",
    "total_amount": "currency",
    "contract_value": "currency",
    "effective_date": "date",
    "expiration_date": "date",
    "party_name": "text",
}

REQUIRED_FIELDS = {
    "invoice": ["invoice_number", "vendor_name", "total_amount"],
    "contract": ["party_name", "effective_date"],
}

# COMMAND ----------

sample_raw_extractions = [
    {
        "file_name": "invoice_sample_01.pdf",
        "doc_type": "invoice",
        "raw_fields": {
            "invoice_number": "  INV-2026-001  ",
            "vendor_name": "Acme   Corp",
            "invoice_date": "03/15/2026",
            "due_date": "April 15, 2026",
            "total_amount": "$12,345.67",
        },
    },
    {
        "file_name": "contract_sample_01.pdf",
        "doc_type": "contract",
        "raw_fields": {
            "party_name": "  Widget Industries LLC  ",
            "effective_date": "2026-01-01",
            "expiration_date": "12/31/2026",
            "contract_value": "$500,000.00",
        },
    },
]

# COMMAND ----------

postprocessed_results = []

for extraction in sample_raw_extractions:
    print(f"\nPostprocessing: {extraction['file_name']}")
    print(f"  Raw: {extraction['raw_fields']}")

    normalized = normalize_fields(extraction["raw_fields"], FIELD_TYPES)
    print(f"  Normalized: {normalized}")

    doc_type = extraction["doc_type"]
    required = REQUIRED_FIELDS.get(doc_type, [])
    validated, errors = validate_extraction(normalized, required)

    if errors:
        print(f"  Validation errors: {errors}")
    else:
        print(f"  Validation: PASSED")

    postprocessed_results.append({
        "file_name": extraction["file_name"],
        "doc_type": doc_type,
        "normalized_fields": validated,
        "validation_errors": errors,
    })

# COMMAND ----------

import pandas as pd

flat_results = []
for result in postprocessed_results:
    flat = {
        "file_name": result["file_name"],
        "doc_type": result["doc_type"],
        "error_count": len(result["validation_errors"]),
    }
    flat.update(result["normalized_fields"])
    flat_results.append(flat)

display(pd.DataFrame(flat_results))
