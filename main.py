import argparse
from pathlib import Path

from commands.chunk import chunk
from commands.eval import eval_models
from commands.index import index
from commands.prompt import prompt


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompt_parser = subparsers.add_parser("prompt")
    prompt_parser.add_argument("prompt")

    index_parser = subparsers.add_parser("index")
    index_source = index_parser.add_mutually_exclusive_group(required=True)
    index_source.add_argument(
        "path", nargs="?", type=Path, help="Path to the textbook to be indexed"
    )
    index_source.add_argument(
        "--chunks",
        type=Path,
        help="Path to a chunks JSONL (from the chunk command) to index",
    )

    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument(
        "path", type=Path, help="Path to the textbook to be chunked"
    )
    chunk_parser.add_argument(
        "--out",
        type=Path,
        default=Path("chunks.jsonl"),
        help="Output JSONL path (default: chunks.jsonl)",
    )

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("chunks.jsonl"),
        help="Path to JSONL file containing all chunks (default: chunks.jsonl)",
    )
    eval_parser.add_argument(
        "--test-set",
        type=Path,
        required=True,
        help="Path to JSONL file containing questions and gold chunk IDs",
    )
    eval_parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    if args.command == "prompt":
        prompt(args.prompt)
    elif args.command == "index":
        index(args.path, args.chunks)
    elif args.command == "chunk":
        chunk(args.path, args.out)
    elif args.command == "eval":
        eval_models(args.chunks, args.test_set, args.batch_size)


if __name__ == "__main__":
    main()
