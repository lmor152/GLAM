from pathlib import Path
from typing import final, override

from glam.matching.base_matcher import BaseMatcher
from glam.matching.fuzzy import predict
from glam.parsing.parsed_address import ParsedAddresses
from glam.types import NZSA, Addresses, AddressIDs, Confidences


@final
class FuzzyMatcher(BaseMatcher):
    requires_parser = True
    type = "Fuzzy"

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir)

    @override
    def build_dependencies(
        self,
    ) -> None:
        super().build_dependencies()

    def load_dependencies(self) -> None:
        pass

    def check_build(self) -> bool:
        return True  # no builds are required for fuzzy matching

    @override
    def _match_unparsed_addresses(
        self, addresses: Addresses
    ) -> tuple[AddressIDs, Confidences]:
        raise NotImplementedError("Fuzzy matcher requires parsed addresses")

    @override
    def _match_parsed_addresses(
        self, addresses: ParsedAddresses
    ) -> tuple[AddressIDs, Confidences]:
        return predict.lookup_addresses(addresses, NZSA.df)
