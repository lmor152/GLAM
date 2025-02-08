from collections.abc import Sequence
from typing import final, override

from glam.parsing.parsed_address import ParsedAddress
from glam.types import Address, LINZAddress


@final
class MatchedAddress:
    __slots__: list[str] = [
        "search_address",
        "matched_address",
        "confidence",
        "parsed_address",
    ]

    def __init__(
        self,
        /,
        search_address: Address | ParsedAddress,
        matched_address: LINZAddress | None,
        confidence: float,
        parsed_address: ParsedAddress | None = None,
    ) -> None:
        self.search_address = search_address
        self.parsed_address = parsed_address
        self.matched_address = matched_address
        self.confidence = confidence

    @override
    def __repr__(self) -> str:
        if isinstance(self.search_address, ParsedAddress):
            search_address = self.search_address.format_address()
        else:
            search_address = self.search_address

        parsed_part = (
            f"-> parsed to {self.parsed_address} "
            if self.parsed_address is not None
            else ""
        )

        return (
            f"Search address {search_address} "
            f"{parsed_part}"
            f"-> matched to {self.matched_address} "
            f"with {self.confidence} confidence"
        )



MatchedAddresses = Sequence[MatchedAddress]
