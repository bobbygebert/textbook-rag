import argparse
import json
import time
from collections import Counter
from collections.abc import Iterator
from typing import Any, cast

import chromadb
import langchain_text_splitters
import pymupdf
import pymupdf4llm
import tqdm
from chromadb import Metadata
from langchain_core.documents import Document
from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequestMessage
from rich.console import Console
from rich.text import Text
from sentence_transformers import CrossEncoder, SentenceTransformer

LLM_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
LLM_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

N_CANDIDATES = 20
TOP_K = 4

console = Console()


def header(text: str, color: str = "light_sky_blue1"):
    console.print(f"\n{text}", style=f"bold {color}")


def stream_response(llm: Llama, system_prompt: str, user_prompt: str):
    """Stream a chat completion for the given system/user prompt, with timing."""
    messages: list[ChatCompletionRequestMessage] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    start = time.time()
    stream = llm.create_chat_completion(
        messages=messages,
        temperature=0.8,
        max_tokens=512,
        stream=True,
    )
    assert isinstance(stream, Iterator)
    for chunk in stream:
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="", flush=True)
    console.print(f"\n\nTime to generate: {time.time() - start:.0f}s", style="dim")


def prompt(args: argparse.Namespace):
    prompt: str = args.prompt

    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="textbook")
    assert (
        collection.count() > 0
    ), "No collection found under chroma_db. Run the index command to create it."
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = Llama.from_pretrained(
        repo_id=LLM_REPO_ID,
        filename=LLM_FILE,
        n_ctx=4096,
        n_threads=None,
        verbose=False,
    )

    query_embedding = embedding_model.encode(prompt).tolist()
    hits = collection.query([query_embedding], n_results=N_CANDIDATES)
    documents = hits["documents"]
    metadatas = hits["metadatas"]
    assert documents is not None and metadatas is not None
    candidates = documents[0]
    metadata = metadatas[0]

    ranked = reranker.rank(prompt, candidates, top_k=TOP_K, return_documents=True)
    context = "\n\n---\n\n".join(
        f"[page {metadata[int(r['corpus_id'])]['page']}]\n{r['text']}" for r in ranked
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following "
        "pieces of retrieved context to answer the question. If you don't know "
        "the answer, just say that you don't know.\n\n"
        f"Context:\n\n{context}"
    )

    header("System", color="light_pink1")
    system_text = Text(system_prompt)
    system_text.highlight_regex(r"\[page \d+\]", "pale_green1")
    console.print(system_text)
    header("Prompt", color="light_pink1")
    print(prompt)

    header("Response with context")
    stream_response(llm, system_prompt, prompt)

    header("Response without context")
    stream_response(llm, "", prompt)

    llm.close()


def chunk_document(doc_path: str) -> list[Document]:
    # Parse the PDF to markdown, one entry per page (with page-number metadata).
    # header/footer=False drops running heads/footers so they don't pollute chunks.
    doc = pymupdf.open(doc_path)
    pages = cast(
        list[dict[str, Any]],
        pymupdf4llm.to_markdown(
            doc=doc,
            footer=False,
            header=False,
            show_progress=True,
            use_ocr=False,
            write_images=False,
            page_chunks=True,
        ),
    )

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    markdown_splitter = langchain_text_splitters.MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    text_splitter = langchain_text_splitters.RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    final_chunks = []
    for page in pages:
        page_number = page["metadata"]["page_number"]
        md_chunks = markdown_splitter.split_text(page["text"])
        page_chunks = text_splitter.split_documents(md_chunks)
        for c in page_chunks:
            c.metadata["page"] = page_number
        final_chunks.extend(page_chunks)
    return final_chunks


def chunk(args: argparse.Namespace):
    doc_path: str = args.path
    out_path: str = args.out

    final_chunks = chunk_document(doc_path)

    per_page_count = Counter()
    with open(out_path, "w") as f:
        for chunk in final_chunks:
            p = chunk.metadata["page"]
            c = per_page_count[p]
            per_page_count[p] += 1
            record = {"id": f"p{p}_c{c}", "page": p, "text": chunk.page_content}
            f.write(json.dumps(record) + "\n")
    print(f"chunks: {len(final_chunks)} -> {out_path}")


def load_chunks(chunks_path: str) -> list[Document]:
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


def index(args: argparse.Namespace):
    doc_path: str | None = args.path
    chunks_path: str | None = args.chunks
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="textbook")
    assert (
        collection.count() == 0
    ), "Remove existing collection under chroma_db if you wish to replace it."
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # argparse guarantees exactly one of path / --chunks is set.
    if chunks_path:
        final_chunks = load_chunks(chunks_path)
    else:
        assert doc_path is not None
        final_chunks = chunk_document(doc_path)

    embeddings = []
    for c in tqdm.tqdm(final_chunks):
        embeddings.append(embedding_model.encode(c.page_content))

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


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompt_parser = subparsers.add_parser("prompt")
    prompt_parser.add_argument("prompt")
    prompt_parser.set_defaults(func=prompt)

    index_parser = subparsers.add_parser("index")
    index_source = index_parser.add_mutually_exclusive_group(required=True)
    index_source.add_argument(
        "path", nargs="?", help="Path to the textbook to be indexed"
    )
    index_source.add_argument(
        "--chunks", help="Path to a chunks JSONL (from the chunk command) to index"
    )
    index_parser.set_defaults(func=index)

    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("path", help="Path to the textbook to be chunked")
    chunk_parser.add_argument(
        "--out",
        default="chunks.jsonl",
        help="Output JSONL path (default: chunks.jsonl)",
    )
    chunk_parser.set_defaults(func=chunk)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
