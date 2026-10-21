"""LSTM with self-attention for text classification on SST."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Additive attention mechanism over sequence time steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # lstm_output: [batch_size, seq_len, hidden_dim]
        energy = self.projection(lstm_output).squeeze(2)  # [batch, seq_len]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)
        weights = F.softmax(energy, dim=1).unsqueeze(2)  # [batch, seq_len, 1]
        context = torch.sum(lstm_output * weights, dim=1)  # [batch, hidden_dim]
        return context


class LSTMAttention(nn.Module):
    """LSTM with word embeddings and attention for sentiment analysis."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 100,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.lstm_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = Attention(self.lstm_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.lstm_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        mask = (x != 0).long()
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embedded)
        context = self.attention(lstm_out, mask)
        context = self.dropout(context)
        logits = self.fc(context)
        return logits
