from pathlib import Path

from glam.matching.base_matcher import BaseMatcher
from glam.matching.hybrid_tfidf import predict
from glam.matching.tfidf.tfidf_matcher import TFIDFMatcher
from glam.matching.vector.vector_matcher import VectorMatcher
from glam.parsing.parsed_address import ParsedAddresses
from glam.types import Addresses, AddressIDs, Confidences


class HybridTFIDFMatcher(BaseMatcher):
    type = "HybridTFIDF"
    requires_parser = True

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir)

        self.tfidf_matcher = TFIDFMatcher(data_dir)
        self.vector_matcher = VectorMatcher(data_dir)

    def check_build(self) -> bool:
        return self.tfidf_matcher.check_build() and self.vector_matcher.check_build()

    def build_dependencies(self) -> None:
        self.tfidf_matcher.build_dependencies()
        self.vector_matcher.build_dependencies()

    def load_dependencies(self) -> None:
        self.tfidf_matcher.load_dependencies()
        self.vector_matcher.load_dependencies()

    def _match_unparsed_addresses(
        self, addresses: Addresses
    ) -> tuple[AddressIDs, Confidences]:
        raise NotImplementedError

    def _match_parsed_addresses(
        self, addresses: ParsedAddresses, **kwargs: object
    ) -> tuple[AddressIDs, Confidences]:
        neighbours = predict.nearest_neighbours(
            addresses,
            self.vector_matcher.trees,
            self.tfidf_matcher.vectoriser,
            self.tfidf_matcher.tfidf_matrix,
            self.vector_matcher.idx_map,
            **kwargs,
        )

        return neighbours["address_id"].fillna(-1).to_list(), neighbours[  # type: ignore
            "confidence"
        ].to_list()
