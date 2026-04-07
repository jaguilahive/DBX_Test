import time
import logging

logger = logging.getLogger("extraction_pipeline")

DEFAULT_EMBEDDING_ENDPOINT = "databricks-bge-large-en"
DEFAULT_BATCH_SIZE = 16
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


def get_embedding_client():
    from databricks.sdk import WorkspaceClient
    workspace_client = WorkspaceClient()
    return workspace_client


def generate_embeddings_batch(texts, endpoint_name=None, batch_size=None):
    endpoint = endpoint_name or DEFAULT_EMBEDDING_ENDPOINT
    batch = batch_size or DEFAULT_BATCH_SIZE

    all_embeddings = []

    for start_index in range(0, len(texts), batch):
        batch_texts = texts[start_index:start_index + batch]
        batch_embeddings = _call_embedding_endpoint(batch_texts, endpoint)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def _call_embedding_endpoint(texts, endpoint_name):
    from databricks.sdk import WorkspaceClient

    workspace_client = WorkspaceClient()

    for attempt in range(MAX_RETRIES):
        try:
            response = workspace_client.serving_endpoints.query(
                name=endpoint_name,
                input=texts,
            )
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    f"Embedding request failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(exc)}"
                )
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
            else:
                raise


def generate_single_embedding(text, endpoint_name=None):
    result = generate_embeddings_batch([text], endpoint_name=endpoint_name)
    return result[0]


def embed_chunks(chunks, endpoint_name=None, batch_size=None):
    texts = [chunk.text for chunk in chunks]
    embeddings = generate_embeddings_batch(
        texts,
        endpoint_name=endpoint_name,
        batch_size=batch_size,
    )
    return list(zip(chunks, embeddings))
