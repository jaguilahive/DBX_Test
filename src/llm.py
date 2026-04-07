import json
import time
import logging

logger = logging.getLogger("extraction_pipeline")

DEFAULT_MODEL_ENDPOINT = "databricks-meta-llama-3-1-70b-instruct"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def call_llm(prompt, system_prompt=None, endpoint_name=None, max_tokens=None, temperature=None):
    endpoint = endpoint_name or DEFAULT_MODEL_ENDPOINT
    max_tok = max_tokens or DEFAULT_MAX_TOKENS
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return _call_serving_endpoint(messages, endpoint, max_tok, temp)


def call_llm_json(prompt, system_prompt=None, endpoint_name=None, max_tokens=None):
    raw_response = call_llm(
        prompt,
        system_prompt=system_prompt,
        endpoint_name=endpoint_name,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    return _parse_json_response(raw_response)


def _call_serving_endpoint(messages, endpoint_name, max_tokens, temperature):
    from databricks.sdk import WorkspaceClient

    workspace_client = WorkspaceClient()

    for attempt in range(MAX_RETRIES):
        try:
            response = workspace_client.serving_endpoints.query(
                name=endpoint_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(exc)}"
                )
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
            else:
                raise


def _parse_json_response(raw_text):
    cleaned = raw_text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse JSON from LLM response: {str(exc)}")
        logger.error(f"Raw response: {raw_text[:500]}")
        raise


def build_extraction_prompt(text, schema_fields):
    field_descriptions = "\n".join(
        f"- {field_name}: {field_type}"
        for field_name, field_type in schema_fields.items()
    )

    prompt = (
        f"Extract the following fields from the document text below. "
        f"Return a JSON object with these fields:\n\n"
        f"{field_descriptions}\n\n"
        f"If a field cannot be found, set its value to null.\n\n"
        f"Document text:\n{text}"
    )
    return prompt


def build_qa_prompt(question, context_chunks):
    context_text = "\n\n---\n\n".join([
        f"[Source: {chunk.file_name}, Page {chunk.start_page}]\n{chunk.text}"
        for chunk in context_chunks
    ])

    prompt = (
        f"Answer the following question based ONLY on the provided context. "
        f"If the context does not contain enough information, say so. "
        f"Cite the source document and page number for each claim.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}"
    )
    return prompt
