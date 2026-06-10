from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    query_prefix: str = ""
    doc_prefix: str = ""


MINILM = EmbeddingModel("all-MiniLM-L6-v2")
BGE_SMALL = EmbeddingModel(
    "BAAI/bge-small-en-v1.5",
    query_prefix="Represent this sentence for searching relevant passages: ",
)
E5_SMALL = EmbeddingModel(
    "intfloat/e5-small-v2",
    query_prefix="query: ",
    doc_prefix="passage: ",
)
BGE_BASE = EmbeddingModel(
    "BAAI/bge-base-en-v1.5",
    query_prefix="Represent this sentence for searching relevant passages: ",
)

EMBEDDING_MODEL = E5_SMALL

RERANKING_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
LLM_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
