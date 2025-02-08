from typing import final, override

import torch.nn as nn
from torch import Tensor

from glam.parsing.rnn.dataset import n_labels, vocab

# Hyperparameters
VOCAB_SIZE = len(vocab) + 1
EMBEDDING_DIM = 24
HIDDEN_DIM = 64
OUTPUT_DIM = n_labels
N_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 128
NUM_EPOCHS = 2


@final
class AddressParser(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # GRU Layer
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.fc2 = nn.Linear(32, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    @override
    def forward(self, text: Tensor, mask: Tensor | None = None) -> Tensor:
        # text.shape = [ batch_size, seq_len ]

        embedded = self.embedding(text)

        output, _ = self.rnn(embedded)

        output = self.dropout(output)

        output = self.fc1(output)
        output = self.fc2(output)
        # output shape = [ batch_size, seq_len, output_dim ]

        if mask is not None:
            output = output * mask.unsqueeze(2)

        return output


address_parser_model = AddressParser(
    VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT
)
