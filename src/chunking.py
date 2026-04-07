import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from schemas import DocumentChunk, ParsedDocument, generate_chunk_id, table_to_markdown


class DocumentChunker:

    def __init__(self, chunk_size=512, chunk_overlap=64, min_chunk_size=50, encoding_name="cl100k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.encoding_name = encoding_name
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            import tiktoken
            self._encoder = tiktoken.get_encoding(self.encoding_name)
        return self._encoder

    def chunk_document(self, doc):
        if not doc.is_successful or not doc.extracted_text.strip():
            return []

        chunks = []
        chunk_index = 0

        chunk_index = self._chunk_narrative_text(doc, chunks, chunk_index)
        chunk_index = self._chunk_tables(doc, chunks, chunk_index)

        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def _chunk_narrative_text(self, doc, chunks, chunk_index):
        page_tokens = []
        for page_info in doc.pages:
            page_num = page_info["page_number"]
            page_text = page_info["text"]
            tokens = self.encoder.encode(page_text)
            for token in tokens:
                page_tokens.append((token, page_num))

        if not page_tokens:
            return chunk_index

        step = self.chunk_size - self.chunk_overlap
        position = 0

        while position < len(page_tokens):
            window = page_tokens[position:position + self.chunk_size]
            token_ids = [t[0] for t in window]
            page_numbers = [t[1] for t in window]

            if len(token_ids) < self.min_chunk_size and position > 0:
                break

            text = self.encoder.decode(token_ids)

            chunks.append(DocumentChunk(
                chunk_id=generate_chunk_id(doc.doc_id, chunk_index),
                doc_id=doc.doc_id,
                chunk_index=chunk_index,
                total_chunks=0,
                text=text,
                token_count=len(token_ids),
                char_count=len(text),
                start_page=min(page_numbers),
                end_page=max(page_numbers),
                source_path=doc.source_path,
                file_name=doc.file_name,
                file_type=doc.file_type,
                metadata=dict(doc.metadata),
                contains_table=False,
                extraction_timestamp=doc.extraction_timestamp,
            ))

            chunk_index += 1
            position += step

        return chunk_index

    def _chunk_tables(self, doc, chunks, chunk_index):
        for table in doc.tables:
            table_text = table_to_markdown(table)
            token_ids = self.encoder.encode(table_text)

            table_meta = {
                **doc.metadata,
                "table_id": table.table_id,
                "sheet_name": table.sheet_name or "",
            }

            if len(token_ids) <= self.chunk_size:
                chunks.append(DocumentChunk(
                    chunk_id=generate_chunk_id(doc.doc_id, chunk_index),
                    doc_id=doc.doc_id,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    text=table_text,
                    token_count=len(token_ids),
                    char_count=len(table_text),
                    start_page=table.page_number,
                    end_page=table.page_number,
                    source_path=doc.source_path,
                    file_name=doc.file_name,
                    file_type=doc.file_type,
                    metadata=table_meta,
                    contains_table=True,
                    extraction_timestamp=doc.extraction_timestamp,
                ))
                chunk_index += 1
            else:
                chunk_index = self._chunk_large_table(
                    doc, table, table_meta, chunks, chunk_index,
                )

        return chunk_index

    def _chunk_large_table(self, doc, table, table_meta, chunks, chunk_index):
        header_text = "| " + " | ".join(table.headers) + " |\n"
        header_text += "| " + " | ".join(["---"] * len(table.headers)) + " |\n"
        header_tokens = len(self.encoder.encode(header_text))

        current_rows_text = header_text
        current_token_count = header_tokens

        for row in table.rows:
            row_text = "| " + " | ".join(row) + " |\n"
            row_tokens = len(self.encoder.encode(row_text))

            if current_token_count + row_tokens > self.chunk_size and current_token_count > header_tokens:
                chunks.append(DocumentChunk(
                    chunk_id=generate_chunk_id(doc.doc_id, chunk_index),
                    doc_id=doc.doc_id,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    text=current_rows_text,
                    token_count=current_token_count,
                    char_count=len(current_rows_text),
                    start_page=table.page_number,
                    end_page=table.page_number,
                    source_path=doc.source_path,
                    file_name=doc.file_name,
                    file_type=doc.file_type,
                    metadata=table_meta,
                    contains_table=True,
                    extraction_timestamp=doc.extraction_timestamp,
                ))
                chunk_index += 1
                current_rows_text = header_text
                current_token_count = header_tokens

            current_rows_text += row_text
            current_token_count += row_tokens

        if current_token_count > header_tokens:
            chunks.append(DocumentChunk(
                chunk_id=generate_chunk_id(doc.doc_id, chunk_index),
                doc_id=doc.doc_id,
                chunk_index=chunk_index,
                total_chunks=0,
                text=current_rows_text,
                token_count=current_token_count,
                char_count=len(current_rows_text),
                start_page=table.page_number,
                end_page=table.page_number,
                source_path=doc.source_path,
                file_name=doc.file_name,
                file_type=doc.file_type,
                metadata=table_meta,
                contains_table=True,
                extraction_timestamp=doc.extraction_timestamp,
            ))
            chunk_index += 1

        return chunk_index
