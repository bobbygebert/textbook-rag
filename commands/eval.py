import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from commands.models import EMBEDDING_MODEL

console = Console()

RECALL_KS = (1, 5, 20)


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    query_prefix: str = ""
    doc_prefix: str = ""


EVAL_MODELS = [
    EmbeddingModel(EMBEDDING_MODEL),
    EmbeddingModel(
        "BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    EmbeddingModel(
        "intfloat/e5-small-v2",
        query_prefix="query: ",
        doc_prefix="passage: ",
    ),
    EmbeddingModel(
        "BAAI/bge-base-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
]


def encode(
    model: SentenceTransformer, texts: list[str], prefix: str, batch_size: int
) -> np.ndarray:
    inputs = [prefix + t for t in texts] if prefix else texts
    return cast(
        np.ndarray,
        model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
    )


def eval_models(chunks_path: Path, test_set_path: Path, batch_size: int):
    ids: list[str] = []
    texts: list[str] = []
    with open(chunks_path) as f:
        for line in f:
            record = json.loads(line)
            ids.append(record["id"])
            texts.append(record["text"])
    id_to_idx = {cid: i for i, cid in enumerate(ids)}

    questions: list[str] = []
    gold_indices: list[int] = []
    with open(test_set_path) as f:
        for line in f:
            record = json.loads(line)
            gold_id = record["gold_id"]
            assert (
                gold_id in id_to_idx
            ), f"gold id {gold_id!r} not found in {chunks_path}"
            questions.append(record["question"])
            gold_indices.append(id_to_idx[gold_id])

    n = len(questions)

    title = f"Embedding model retrieval (recall@k, n={n})"
    table = Table(title=title)
    table.add_column("model")
    for k in RECALL_KS:
        table.add_column(f"recall@{k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("embed_s", justify="right")
    table.add_column("chunks/s", justify="right")

    for cfg in EVAL_MODELS:
        console.print(f"\n{cfg.name}", style="bold")
        model = SentenceTransformer(cfg.name)
        start = time.time()
        chunk_embeddings = encode(model, texts, cfg.doc_prefix, batch_size)
        embed_time = time.time() - start
        query_embeddings = encode(model, questions, cfg.query_prefix, batch_size)
        sims = query_embeddings @ chunk_embeddings.T
        gold_sims = sims[np.arange(n), np.array(gold_indices)]
        ranks = (sims > gold_sims.reshape(n, 1)).sum(axis=1) + 1
        recalls = [int((ranks <= k).sum()) / n for k in RECALL_KS]
        mrr = float((1.0 / ranks).mean())
        chunks_per_s = len(texts) / embed_time
        table.add_row(
            cfg.name,
            *[f"{r:.3f}" for r in recalls],
            f"{mrr:.3f}",
            f"{embed_time:.1f}",
            f"{chunks_per_s:.0f}",
        )

    console.print(table)
