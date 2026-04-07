import json
import os
import logging

logger = logging.getLogger("extraction_pipeline")


def compute_completeness_score(extracted_text, ground_truth_text):
    ground_truth_tokens = set(ground_truth_text.lower().split())
    extracted_tokens = set(extracted_text.lower().split())
    if not ground_truth_tokens:
        return 0.0
    return len(ground_truth_tokens & extracted_tokens) / len(ground_truth_tokens)


def compute_accuracy_score(extracted_text, ground_truth_text):
    ground_truth_tokens = set(ground_truth_text.lower().split())
    extracted_tokens = set(extracted_text.lower().split())
    union_count = len(ground_truth_tokens | extracted_tokens)
    if union_count == 0:
        return 0.0
    return len(ground_truth_tokens & extracted_tokens) / union_count


def compute_field_accuracy(extracted_fields, ground_truth_fields):
    if not ground_truth_fields:
        return 0.0

    correct_count = 0
    total_fields = len(ground_truth_fields)

    for field_name, expected_value in ground_truth_fields.items():
        actual_value = extracted_fields.get(field_name)
        if actual_value is None:
            continue
        if str(actual_value).strip().lower() == str(expected_value).strip().lower():
            correct_count += 1

    return correct_count / total_fields


def compute_field_completeness(extracted_fields, ground_truth_fields):
    if not ground_truth_fields:
        return 0.0

    present_count = sum(
        1 for field_name in ground_truth_fields
        if extracted_fields.get(field_name) is not None
        and str(extracted_fields.get(field_name)).strip() != ""
    )

    return present_count / len(ground_truth_fields)


def compute_table_preservation_score(extracted_table, ground_truth_table):
    if not ground_truth_table or not extracted_table:
        return 0.0

    gt_cells = set()
    for row_idx, row in enumerate(ground_truth_table.get("rows", [])):
        for col_idx, cell in enumerate(row):
            gt_cells.add((row_idx, col_idx, str(cell).strip().lower()))

    ext_cells = set()
    for row_idx, row in enumerate(extracted_table.get("rows", [])):
        for col_idx, cell in enumerate(row):
            ext_cells.add((row_idx, col_idx, str(cell).strip().lower()))

    if not gt_cells:
        return 0.0

    true_positives = len(gt_cells & ext_cells)
    precision = true_positives / max(len(ext_cells), 1)
    recall = true_positives / len(gt_cells)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def evaluate_extraction_batch(documents, ground_truth_dir):
    results = []

    for doc in documents:
        gt_filename = os.path.splitext(doc.file_name)[0] + ".json"
        gt_path = os.path.join(ground_truth_dir, gt_filename)

        if not os.path.exists(gt_path):
            logger.warning(f"No ground truth found for {doc.file_name}")
            continue

        with open(gt_path, "r") as fh:
            ground_truth = json.load(fh)

        gt_text = ground_truth.get("expected_text", "")
        completeness = compute_completeness_score(doc.extracted_text, gt_text)
        accuracy = compute_accuracy_score(doc.extracted_text, gt_text)

        gt_fields = ground_truth.get("expected_fields", {})
        field_accuracy = 0.0
        field_completeness = 0.0
        if gt_fields and hasattr(doc, "extracted_fields"):
            field_accuracy = compute_field_accuracy(doc.extracted_fields, gt_fields)
            field_completeness = compute_field_completeness(doc.extracted_fields, gt_fields)

        result = {
            "doc_id": doc.doc_id,
            "file_name": doc.file_name,
            "completeness_score": round(completeness, 4),
            "accuracy_score": round(accuracy, 4),
            "field_accuracy": round(field_accuracy, 4),
            "field_completeness": round(field_completeness, 4),
            "extraction_quality_score": doc.extraction_quality_score,
            "page_count": doc.page_count,
            "table_count": doc.table_count,
            "error_count": len(doc.errors),
            "is_successful": doc.is_successful,
        }
        results.append(result)

    return results


def compute_aggregate_metrics(evaluation_results):
    if not evaluation_results:
        return {}

    count = len(evaluation_results)
    avg_completeness = sum(r["completeness_score"] for r in evaluation_results) / count
    avg_accuracy = sum(r["accuracy_score"] for r in evaluation_results) / count
    avg_quality = sum(r["extraction_quality_score"] for r in evaluation_results) / count
    total_errors = sum(r["error_count"] for r in evaluation_results)
    success_count = sum(1 for r in evaluation_results if r["is_successful"])
    failure_rate = 1.0 - (success_count / count)

    return {
        "document_count": count,
        "average_completeness": round(avg_completeness, 4),
        "average_accuracy": round(avg_accuracy, 4),
        "average_quality_score": round(avg_quality, 4),
        "total_error_count": total_errors,
        "success_rate": round(success_count / count, 4),
        "failure_rate": round(failure_rate, 4),
    }


def log_evaluation_to_mlflow(evaluation_results, aggregate_metrics, experiment_path=None):
    import mlflow

    if experiment_path:
        mlflow.set_experiment(experiment_path)

    with mlflow.start_run(run_name="extraction_evaluation") as run:
        for metric_name, metric_value in aggregate_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        for idx, result in enumerate(evaluation_results):
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{key}_doc{idx}", value)

        mlflow.set_tag("evaluation_type", "extraction")
        mlflow.set_tag("document_count", str(len(evaluation_results)))

        return run.info.run_id
