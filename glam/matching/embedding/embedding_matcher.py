import pickle
from pathlib import Path
from typing import override

import pandas as pd
import torch
from scipy.spatial import cKDTree

from glam.logs import get_logger
from glam.matching.base_matcher import BaseMatcher
from glam.matching.embedding.model import (
    TransformerModel,
    infer,
)
from glam.matching.embedding.predict import nearest_neighbours
from glam.parsing.parsed_address import ParsedAddresses
from glam.types import NZSA, Addresses, AddressIDs, Confidences

# Hyperparameters
EMBEDDING_DIM = 16
NUM_HEADS = 4
HIDDEN_DIM = 128
OUTPUT_DIM = 128
N_LAYERS = 4


class EmbeddingMatcher(BaseMatcher):
    type = "Embedding"
    requires_parser = False
    model: TransformerModel
    tree: cKDTree
    idx_map: dict[int, int]

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir)
        self.device = self.get_device()
        self.vocab = self.load_vocab(Path(__file__).parent / "vocab.csv")
        self.model = self.load_model(Path(__file__).parent / "model.pth")

    @override
    def check_build(self) -> bool:
        vectors_check = self.matcher_data / "idx_map.csv"
        return vectors_check.exists()

    @override
    def build_dependencies(self) -> None:
        super().build_dependencies()

        if NZSA.postcodes:
            addresses = NZSA.df["full_address_ascii"] + " " + NZSA.df["postcode"]
        else:
            addresses = NZSA.df["full_address_ascii"]

        # needs to be at least as long as the lenght of n-grams
        addresses = addresses.fillna("__")

        embeddings = infer(self.model, addresses, self.device, self.vocab)

        tree = cKDTree(embeddings)
        idx_map = {
            idx: address_id for idx, address_id in enumerate(NZSA.df["address_id"])
        }

        tree_path = self.matcher_data / "tree.pkl"
        with open(tree_path, "wb") as f:
            pickle.dump(tree, f)

        idx_map_path = self.matcher_data / "idx_map.csv"
        pd.DataFrame(idx_map.items(), columns=["idx", "address_id"]).to_csv(
            idx_map_path, index=False
        )

    @override
    def load_dependencies(self) -> None:
        idx_map_path = self.matcher_data / "idx_map.csv"
        tree_path = self.matcher_data / "tree.pkl"
        self.idx_map = pd.read_csv(idx_map_path, index_col="idx")[
            "address_id"
        ].to_dict()
        with open(tree_path, "rb") as f:
            self.tree: cKDTree = pickle.load(f)

    @override
    def _match_parsed_addresses(
        self, addresses: ParsedAddresses
    ) -> tuple[AddressIDs, Confidences]:
        raise NotImplementedError("VectorMatcher does not support parsed addresses")

    @override
    def _match_unparsed_addresses(
        self, addresses: Addresses
    ) -> tuple[AddressIDs, Confidences]:
        embeddings = infer(self.model, addresses, self.device, self.vocab)

        matches, distances = nearest_neighbours(embeddings, self.tree)

        # Convert euclidean distance to cossim
        distances = 100 * (1 - (distances**2) / 2)
        matches = [self.idx_map[idx] for idx in matches]

        return matches, distances

    def get_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")

        # Disabling MPS for now as masking
        # is not supported for transformer encoder layers
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        else:
            device = torch.device("cpu")

        logger = get_logger()
        logger.info(f"Using device {device}")
        return device

    def load_vocab(self, vocab_path: Path) -> list[str]:
        with open(vocab_path, "r") as f:
            return f.read().splitlines()

    def load_model(self, model_path: Path) -> TransformerModel:
        address_embedding_model = TransformerModel(
            vocab_size=len(self.vocab),
            embed_size=EMBEDDING_DIM,
            num_heads=NUM_HEADS,
            hidden_dim=HIDDEN_DIM,
            num_layers=N_LAYERS,
            output_dim=OUTPUT_DIM,
        )

        _ = address_embedding_model.to(self.device)
        _ = address_embedding_model.load_state_dict(
            torch.load(model_path, weights_only=True)  # pyright: ignore[reportUnknownMemberType, reportAny]
        )
        address_embedding_model.eval()
        return address_embedding_model
