# textbook-rag

## Installing uv

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the [installation docs](https://docs.astral.sh/uv/getting-started/installation/) for other methods.

## Usage

Index a PDF:

```sh
uv run main.py index path/to/textbook.pdf
```

Ask a question:

```sh
uv run main.py prompt "What is the chain rule of probability?"
```
