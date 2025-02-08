import pickle
from glob import glob
from pathlib import Path
from typing import override

import pandas as pd
from scipy.spatial import cKDTree

from glam.matching.base_matcher import BaseMatcher
from glam.matching.vector import predict, vectors
from glam.parsing.parsed_address import ParsedAddresses
from glam.types import Addresses, AddressIDs, Confidences


class VectorMatcher(BaseMatcher):
    type = "Vector"
    requires_parser = True
    trees: dict[int, cKDTree]
    idx_map: dict[int, int]

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir)

    @override
    def check_build(self) -> bool:
        vectors_check = self.matcher_data / "idx_map.csv"
        return vectors_check.exists()

    @override
    def build_dependencies(self) -> None:
        super().build_dependencies()

        vectors.build_vectors2(
            self.matcher_data,
        )

    @override
    def load_dependencies(self) -> None:
        files = glob(str(self.matcher_data / "tree*.pkl"))

        self.trees = {}
        for fpath in files:
            with open(fpath, "rb") as f:
                tree: cKDTree = pickle.load(f)
                self.trees[tree.m] = tree

        # maps tree index to address id
        self.idx_map = (
            pd.read_csv(self.matcher_data / "idx_map.csv")  # type: ignore
            .set_index("idx")["address_id"]
            .to_dict()
        )

    @override
    def _match_parsed_addresses(
        self, addresses: ParsedAddresses
    ) -> tuple[AddressIDs, Confidences]:
        neighbours = predict.nearest_neighbours(addresses, self.trees)
        neighbours["address_id"] = neighbours["match"].map(self.idx_map)  # type: ignore

        return neighbours["address_id"].to_list(), neighbours[  # type: ignore
            "embedding_distance"
        ].to_list()

    @override
    def _match_unparsed_addresses(
        self, addresses: Addresses
    ) -> tuple[AddressIDs, Confidences]:
        raise NotImplementedError("VectorMatcher does not support unparsed addresses")
