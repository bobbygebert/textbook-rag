import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import langchain_text_splitters
import pymupdf
import pymupdf4llm
from langchain_core.documents import Document


def chunk_document(doc_path: Path) -> list[Document]:
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


def chunk(doc_path: Path, out_path: Path):
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
