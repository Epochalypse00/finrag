from finrag.ingestion.chunk import chunk_documents
from langchain.schema import Document


def test_chunker_non_empty():
    docs = [Document(page_content="Hello world. " * 200,
                     metadata={"file_name": "a.pdf", "page": 1})]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 0
    assert all(c.page_content.strip() for c in chunks)
