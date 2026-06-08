import argparse
from collections.abc import Iterator
from typing import Any, cast

import chromadb
import langchain_text_splitters
import pymupdf
import pymupdf4llm
import tqdm
from chromadb import Metadata
from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequestMessage
from sentence_transformers import CrossEncoder, SentenceTransformer

LLM_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
LLM_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

N_CANDIDATES = 20
TOP_K = 4


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

    messages: list[ChatCompletionRequestMessage] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
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
    print()


def index(args: argparse.Namespace):
    doc_path: str = args.path
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="textbook")
    assert (
        collection.count() == 0
    ), "Remove existing collection under chroma_db if you wish to replace it."
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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
    index_parser.add_argument("path", help="Path to the textbook to be indexed")
    index_parser.set_defaults(func=index)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
