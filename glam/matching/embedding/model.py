import math
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from glam.types import Addresses


def tokenize(text: str) -> list[str]:
    # Remove punctuation and lowercase
    text = re.sub(r"[^\w\s]", "", text.lower())
    grams = [text[i : i + 2] for i in range(len(text) - 1)]  # bigrams
    # grams = list(text)  # unigrams
    # grams = [text[i:i+3] for i in range(len(text) - 2)] # trigrams
    return grams


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_size)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :].to(x.device)  #


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_heads,
        hidden_dim,
        num_layers,
        output_dim,
        max_len=512,
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size + 2, embed_size, padding_idx=0
        )  # Padding at 0
        self.positional_encoding = PositionalEncoding(embed_size, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(embed_size, output_dim)

    def forward(self, input):
        input_embed = self.embedding(input)  # (batch_size, seq_len, embed_size)

        padding_mask = input == 0  # (batch_size, seq_len)

        input_embed = self.positional_encoding(input_embed)  # Apply positional encoding
        input_out = self.transformer_encoder(
            input_embed, src_key_padding_mask=padding_mask
        )

        # this does mean pooling with masking
        input_out = input_out.masked_fill(
            padding_mask.unsqueeze(-1), 0
        )  # Mask padding embeddings
        input_out = input_out.sum(dim=1) / (~padding_mask).sum(dim=1, keepdim=True)

        output = self.fc(input_out)
        output = F.normalize(output, p=2, dim=1)
        return output


def infer(
    model, addresses: Addresses, device: torch.device, vocab: list[str], batch_size=128
):
    # Tokenize and convert to tensors
    def text_to_tensor(text: str) -> torch.Tensor:
        bigrams = tokenize(text)
        bigram_to_idx = {bigram: idx + 2 for idx, bigram in enumerate(vocab)}
        indices = [bigram_to_idx.get(bigram, 1) for bigram in bigrams]  # OOV handling
        return torch.tensor(indices, dtype=torch.long)

    # Convert list of addresses to tensors
    address_tensors = [text_to_tensor(addr) for addr in addresses]

    # Pad sequences to the same length
    address_tensors_padded = pad_sequence(
        address_tensors, batch_first=True, padding_value=0
    )

    # Create DataLoader for batching
    dataloader = DataLoader(
        address_tensors_padded, batch_size=batch_size, shuffle=False
    )

    embeddings = []
    with torch.no_grad():  # No gradient tracking
        for batch in dataloader:
            batch = batch.to(device)
            batch_embeddings = model(batch)
            embeddings.append(batch_embeddings.cpu().numpy())

    return np.concatenate(embeddings)  # Concatenate all batches
