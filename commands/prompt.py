import time
from collections.abc import Iterator

import chromadb
from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequestMessage
from rich.console import Console
from rich.text import Text
from sentence_transformers import CrossEncoder, SentenceTransformer

from commands.models import EMBEDDING_MODEL, LLM_FILE, LLM_REPO_ID, RERANKING_MODEL

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


def prompt(question: str):
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_or_create_collection(name="textbook")
    assert (
        collection.count() > 0
    ), "No collection found under chroma_db. Run the index command to create it."
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKING_MODEL)
    llm = Llama.from_pretrained(
        repo_id=LLM_REPO_ID,
        filename=LLM_FILE,
        n_ctx=4096,
        n_threads=None,
        verbose=False,
    )

    query_embedding = embedding_model.encode(question).tolist()
    hits = collection.query([query_embedding], n_results=N_CANDIDATES)
    documents = hits["documents"]
    metadatas = hits["metadatas"]
    assert documents is not None and metadatas is not None
    candidates = documents[0]
    metadata = metadatas[0]

    ranked = reranker.rank(question, candidates, top_k=TOP_K, return_documents=True)
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
    print(question)

    header("Response with context")
    stream_response(llm, system_prompt, question)

    header("Response without context")
    stream_response(llm, "", question)

    llm.close()
