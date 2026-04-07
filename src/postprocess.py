import re
import logging
from datetime import datetime

logger = logging.getLogger("extraction_pipeline")

DATE_PATTERNS = [
    (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
    (r"\d{2}/\d{2}/\d{4}", "%m/%d/%Y"),
    (r"\d{2}-\d{2}-\d{4}", "%m-%d-%Y"),
    (r"\w+ \d{1,2}, \d{4}", "%B %d, %Y"),
]


def normalize_date_field(raw_value):
    if not raw_value or not isinstance(raw_value, str):
        return raw_value

    cleaned = raw_value.strip()

    for pattern, date_format in DATE_PATTERNS:
        match = re.search(pattern, cleaned)
        if match:
            try:
                parsed = datetime.strptime(match.group(), date_format)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

    return cleaned


def normalize_currency_field(raw_value):
    if not raw_value or not isinstance(raw_value, str):
        return raw_value

    cleaned = raw_value.strip()
    cleaned = cleaned.replace("$", "").replace(",", "").replace(" ", "")

    try:
        return str(round(float(cleaned), 2))
    except ValueError:
        return raw_value


def normalize_text_field(raw_value):
    if not raw_value or not isinstance(raw_value, str):
        return raw_value

    cleaned = raw_value.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def validate_extraction(extracted_fields, required_fields):
    validation_errors = []
    validated_fields = dict(extracted_fields)

    for field_name in required_fields:
        value = validated_fields.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            validation_errors.append({
                "field": field_name,
                "error": "missing_required_field",
                "message": f"Required field '{field_name}' is missing or empty",
            })

    return validated_fields, validation_errors


def normalize_fields(extracted_fields, field_types):
    normalized = {}

    for field_name, raw_value in extracted_fields.items():
        field_type = field_types.get(field_name, "text")

        if field_type == "date":
            normalized[field_name] = normalize_date_field(raw_value)
        elif field_type == "currency":
            normalized[field_name] = normalize_currency_field(raw_value)
        elif field_type == "text":
            normalized[field_name] = normalize_text_field(raw_value)
        else:
            normalized[field_name] = raw_value

    return normalized


def clean_extracted_text(text):
    if not text:
        return ""

    cleaned = text.replace("\x00", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = cleaned.strip()

    return cleaned
