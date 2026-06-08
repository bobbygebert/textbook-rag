import argparse
from collections.abc import Iterator

from llama_cpp import Llama
from llama_cpp.llama_types import ChatCompletionRequestMessage

LLM_REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
LLM_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"


def prompt(args: argparse.Namespace):
    prompt: str = args.prompt

    llm = Llama.from_pretrained(
        repo_id=LLM_REPO_ID,
        filename=LLM_FILE,
        n_ctx=4096,
        n_threads=None,
        verbose=False
    )

    messages: list[ChatCompletionRequestMessage] = [
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


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompt_parser = subparsers.add_parser("prompt")
    prompt_parser.add_argument("prompt")
    prompt_parser.set_defaults(func=prompt)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
