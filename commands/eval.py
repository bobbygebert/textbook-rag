import json
from pathlib import Path
from typing import cast

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer

from commands.models import EMBEDDING_MODEL

console = Console()


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

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    chunk_embeddings = cast(
        np.ndarray,
        embedding_model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
    )
    query_embeddings = cast(
        np.ndarray,
        embedding_model.encode(
            questions,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
    )
    sims = query_embeddings @ chunk_embeddings.T

    top1 = sims.argmax(axis=1)
    hits = int((top1 == gold_indices).sum())
    recall_at_1 = hits / len(questions)

    console.print(
        f"recall@1: {recall_at_1:.3f}  ({hits}/{len(questions)})  "
        f"model={EMBEDDING_MODEL}",
        style="bold",
        highlight=False,
    )
