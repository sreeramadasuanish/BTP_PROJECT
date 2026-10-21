"""Preprocessing and tokenization utilities for vision and text datasets."""

import re
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class TensorDatasetWrapper(Dataset):
    """Memory-efficient Dataset wrapper with proper channel ordering and CIFAR-10 normalization."""

    def __init__(self, data: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)

        # 1. Fix Dimension Permutation: Ensure Channels-First [N, C, H, W]
        if data.ndim == 4 and data.shape[-1] in (1, 3):  # [N, H, W, C] -> [N, C, H, W]
            data = data.permute(0, 3, 1, 2)

        # 2. Scale pixel values to [0, 1]
        if data.dtype == torch.uint8:
            data = data.float() / 255.0
        elif data.dtype == torch.float64:
            data = data.float()

        # 3. Apply Standard Normalization for CIFAR-10 RGB Images (3 channels)
        if data.ndim == 4 and data.shape[1] == 3:
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
            data = (data - mean) / std

        if targets.dtype != torch.long:
            targets = targets.long()

        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class TextTokenizer:
    """Fast regex tokenizer for English sentiment text."""

    def __init__(self):
        self.pattern = re.compile(r"\w+|[^\w\s]")

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            text = str(text)
        return self.pattern.findall(text.lower())


class Vocabulary:
    """Vocabulary mapping tokens to integer IDs with padding and unknown tokens."""

    def __init__(self, pad_token: str = "<pad>", unk_token: str = "<unk>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx: Dict[str, int] = {pad_token: 0, unk_token: 1}
        self.idx2word: Dict[int, str] = {0: pad_token, 1: unk_token}

    def build_vocab(self, texts: List[str], max_vocab_size: int = 10000, min_freq: int = 1) -> None:
        tokenizer = TextTokenizer()
        freqs: Dict[str, int] = {}
        for text in texts:
            for token in tokenizer.tokenize(text):
                freqs[token] = freqs.get(token, 0) + 1

        sorted_tokens = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
        for token, count in sorted_tokens:
            if count < min_freq:
                break
            if len(self.word2idx) >= max_vocab_size:
                break
            if token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token

    def encode(self, text: str, max_length: int = 64) -> List[int]:
        tokenizer = TextTokenizer()
        tokens = tokenizer.tokenize(text)
        unk_idx = self.word2idx[self.unk_token]
        pad_idx = self.word2idx[self.pad_token]

        indices = [self.word2idx.get(t, unk_idx) for t in tokens[:max_length]]
        if len(indices) < max_length:
            indices += [pad_idx] * (max_length - len(indices))
        return indices

    def __len__(self) -> int:
        return len(self.word2idx)