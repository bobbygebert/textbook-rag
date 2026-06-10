import json
from pathlib import Path

import chromadb
import tqdm
from chromadb import Metadata
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from commands.chunk import chunk_document
from commands.models import EMBEDDING_MODEL


def load_chunks(chunks_path: Path) -> list[Document]:
    chunks = []
    with open(chunks_path) as f:
        for line in f:
            record = json.loads(line)
            chunks.append(
                Document(
                    page_content=record["text"],
                    metadata={"page": record["page"]},
                )
            )
    return chunks


def index(doc_path: Path | None, chunks_path: Path | None):
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="textbook", metadata={"hnsw:space": "cosine"}
    )
    assert (
        collection.count() == 0
    ), "Remove existing collection under chroma_db if you wish to replace it."
    embedding_model = SentenceTransformer(EMBEDDING_MODEL.name)

    # argparse guarantees exactly one of path / --chunks is set.
    if chunks_path:
        final_chunks = load_chunks(chunks_path)
    else:
        assert doc_path is not None
        final_chunks = chunk_document(doc_path)

    embeddings = []
    for c in tqdm.tqdm(final_chunks):
        embeddings.append(
            embedding_model.encode(EMBEDDING_MODEL.doc_prefix + c.page_content)
        )

    ids = [str(x) for x in range(len(final_chunks))]
    documents = [c.page_content for c in final_chunks]
    metadatas: list[Metadata] = [{"page": c.metadata["page"]} for c in final_chunks]
    BATCH_SIZE = 5000
    for i in range(0, len(final_chunks), BATCH_SIZE):
        collection.add(
            ids=ids[i : i + BATCH_SIZE],
            embeddings=embeddings[i : i + BATCH_SIZE],
            documents=documents[i : i + BATCH_SIZE],
            metadatas=metadatas[i : i + BATCH_SIZE],
        )
