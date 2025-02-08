from pathlib import Path
from typing import final, override

from scipy.spatial import cKDTree

from glam.matching.hybrid_fuzzy import predict
from glam.matching.vector.vector_matcher import VectorMatcher
from glam.parsing.parsed_address import ParsedAddresses
from glam.types import AddressIDs, Confidences


@final
class HybridFuzzyMatcher(VectorMatcher):
    type = "HybridFuzzy"
    requires_parser = True
    trees: dict[int, cKDTree]
    idx_map: dict[int, int]

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir)

        # overwrite this to not duplicate the data
        self.matcher_data: Path = self.data_dir / "matching" / "Vector"

    @override
    def _match_parsed_addresses(
        self, addresses: ParsedAddresses, **kwargs: object
    ) -> tuple[AddressIDs, Confidences]:
        neighbours = predict.nearest_neighbours(addresses, self.trees, **kwargs)

        return neighbours["address_id"].fillna(-1).to_list(), neighbours[  # type: ignore
            "confidence"
        ].to_list()
