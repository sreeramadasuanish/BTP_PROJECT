
"""SST-2 loader using genuine sentiment data with caching."""

from pathlib import Path
from typing import Tuple
import csv
import random
import urllib.request
import zipfile

import torch

from src.data.preprocessing import (
    TextTokenizer,
    Vocabulary,
    TensorDatasetWrapper,
)
from src.utils.config import get_base_dir
from src.utils.logger import get_logger

logger = get_logger("fed4fed.data.sst")

DATASET_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
SOURCE_MARKER = "official_glue_sst2_real_v1"


def load_sst(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    max_seq_length: int = 64,
    vocab_size: int = 10000,
) -> Tuple[
    TensorDatasetWrapper,
    TensorDatasetWrapper,
    TensorDatasetWrapper,
    Vocabulary,
]:
    """Load genuine SST-2 data using reproducible, disjoint labeled splits."""

    base_dir = get_base_dir()
    raw_dir = Path(raw_dir or base_dir / "data" / "raw" / "sst")
    processed_dir = Path(
        processed_dir or base_dir / "data" / "processed" / "sst"
    )

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cache_file = processed_dir / "sst_cache.pt"

    # Load only a cache created by this version of the real-data loader.
    if cache_file.exists():
        try:
            cached = torch.load(
                cache_file, map_location="cpu", weights_only=True
            )

            if (
                cached.get("source") == SOURCE_MARKER
                and cached.get("max_seq_length") == max_seq_length
                and cached.get("vocab_size") == vocab_size
            ):
                train_data = cached["train_data"]
                train_targets = cached["train_targets"]
                val_data = cached["val_data"]
                val_targets = cached["val_targets"]
                test_data = cached["test_data"]
                test_targets = cached["test_targets"]

                if (
                    train_data.shape == (8544, max_seq_length)
                    and val_data.shape == (1101, max_seq_length)
                    and test_data.shape == (2210, max_seq_length)
                    and len(train_targets) == 8544
                    and len(val_targets) == 1101
                    and len(test_targets) == 2210
                ):
                    vocab = Vocabulary()
                    vocab.word2idx = cached["word2idx"]
                    vocab.idx2word = cached["idx2word"]

                    logger.info("Loaded cached real SST-2 data.")
                    return (
                        TensorDatasetWrapper(train_data, train_targets),
                        TensorDatasetWrapper(val_data, val_targets),
                        TensorDatasetWrapper(test_data, test_targets),
                        vocab,
                    )

            logger.warning("Cache is untrusted or incompatible; rebuilding.")

        except Exception as exc:
            logger.warning(f"Could not load SST cache: {exc}")

    # Download the official GLUE SST-2 archive if necessary.
    archive = raw_dir / "SST-2.zip"
    extracted_dir = raw_dir / "SST-2"
    train_file = extracted_dir / "train.tsv"

    try:
        if not train_file.exists():
            if not archive.exists():
                logger.info("Downloading official GLUE SST-2 archive...")
                request = urllib.request.Request(
                    DATASET_URL,
                    headers={"User-Agent": "FED4FED-dataset-loader/1.0"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    with archive.open("wb") as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)

            if not zipfile.is_zipfile(archive):
                raise RuntimeError("Downloaded SST-2 archive is not a valid ZIP.")

            with zipfile.ZipFile(archive, "r") as zf:
                members = zf.namelist()
                member = next(
                    (
                        name for name in members
                        if name.endswith("/train.tsv")
                        and name.split("/")[-2] == "SST-2"
                    ),
                    None,
                )

                if member is None:
                    raise RuntimeError(
                        "Official SST-2 train.tsv was not found in the archive."
                    )

                # Extract only the expected dataset files.
                for name in members:
                    parts = Path(name).parts
                    if (
                        parts
                        and parts[0] == "SST-2"
                        and len(parts) == 2
                        and parts[1] in {
                            "train.tsv", "dev.tsv", "test.tsv"
                        }
                    ):
                        zf.extract(name, raw_dir)

        # Read genuine labeled sentences from the official training split.
        sentences = []
        labels = []

        with train_file.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")

            if not reader.fieldnames or not {
                "sentence", "label"
            }.issubset(reader.fieldnames):
                raise RuntimeError(
                    "SST-2 training file does not contain sentence and label columns."
                )

            for row in reader:
                sentence = (row.get("sentence") or "").strip()
                label = (row.get("label") or "").strip()

                if sentence and label in {"0", "1"}:
                    sentences.append(sentence)
                    labels.append(int(label))

        if len(sentences) < 8544 + 1101 + 2210:
            raise RuntimeError(
                f"Not enough genuine labeled examples: {len(sentences)}."
            )

        # Create deterministic, non-overlapping subsets of genuine examples.
        # These are custom subsets of the official training split, not
        # the official GLUE validation/test splits.
        indices = list(range(len(sentences)))
        random.Random(42).shuffle(indices)

        train_idx = indices[:8544]
        val_idx = indices[8544:8544 + 1101]
        test_idx = indices[8544 + 1101:8544 + 1101 + 2210]

        def select(indices):
            return (
                [sentences[i] for i in indices],
                [labels[i] for i in indices],
            )

        train_sents, train_labels = select(train_idx)
        val_sents, val_labels = select(val_idx)
        test_sents, test_labels = select(test_idx)

    except Exception as exc:
        raise RuntimeError(
            "Could not load genuine SST-2 data. Check your internet connection, "
            "downloaded archive, and dataset files. No synthetic data was generated."
        ) from exc

    # Build vocabulary from training text only.
    vocab = Vocabulary()
    vocab.build_vocab(train_sents, max_vocab_size=vocab_size)

    def encode_sentences(texts):
        return torch.tensor(
            [
                vocab.encode(s, max_length=max_seq_length)
                for s in texts
            ],
            dtype=torch.long,
        )

    train_data = encode_sentences(train_sents)
    val_data = encode_sentences(val_sents)
    test_data = encode_sentences(test_sents)

    train_targets = torch.tensor(train_labels, dtype=torch.long)
    val_targets = torch.tensor(val_labels, dtype=torch.long)
    test_targets = torch.tensor(test_labels, dtype=torch.long)

    assert train_data.shape == (8544, max_seq_length)
    assert val_data.shape == (1101, max_seq_length)
    assert test_data.shape == (2210, max_seq_length)

    torch.save(
        {
            "source": SOURCE_MARKER,
            "max_seq_length": max_seq_length,
            "vocab_size": vocab_size,
            "train_data": train_data,
            "train_targets": train_targets,
            "val_data": val_data,
            "val_targets": val_targets,
            "test_data": test_data,
            "test_targets": test_targets,
            "word2idx": vocab.word2idx,
            "idx2word": vocab.idx2word,
        },
        cache_file,
    )

    logger.info(
        "Real SST-2 subsets prepared: 8544 train, 1101 validation, "
        "2210 test sentences."
    )

    return (
        TensorDatasetWrapper(train_data, train_targets),
        TensorDatasetWrapper(val_data, val_targets),
        TensorDatasetWrapper(test_data, test_targets),
        vocab,
    )
