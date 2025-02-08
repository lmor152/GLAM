from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import cast, override

from glam.logs import setup_logger
from glam.parsing.parsed_address import ParsedAddress, ParsedAddresses
from glam.types import Addresses, AddressIDs, Confidences

matching_dir = "matching"


class BaseMatcher(ABC):
    requires_parser: bool
    type: str
    deps_loaded: bool = False
    logger: Logger = setup_logger()

    def __init__(self, data_dir: Path) -> None:
        self.data_dir: Path = data_dir
        self.nzsa_loc: Path = self.data_dir / "nz-street-address.csv"
        self.nzsa_upgraded_loc: Path = self.data_dir / "nzsa_upgraded.csv"
        self.matcher_data: Path = self.data_dir / matching_dir / self.type

    @override
    def __repr__(self) -> str:
        return self.type

    @abstractmethod
    def check_build(self) -> bool:
        pass

    @abstractmethod
    def build_dependencies(self) -> None:
        if not Path.exists(self.data_dir / matching_dir):
            Path.mkdir(self.data_dir / matching_dir)
        if not Path.exists(self.matcher_data):
            Path.mkdir(self.matcher_data)

    @abstractmethod
    def load_dependencies(self) -> None:
        pass

    @abstractmethod
    def _match_parsed_addresses(
        self, addresses: ParsedAddresses, **kwargs: object
    ) -> tuple[AddressIDs, Confidences]:
        pass

    @abstractmethod
    def _match_unparsed_addresses(
        self, addresses: Addresses, **kwargs: object
    ) -> tuple[AddressIDs, Confidences]:
        pass

    def match_addresses(
        self, addresses: ParsedAddresses | Addresses, **kwargs: object
    ) -> tuple[AddressIDs, Confidences]:
        n_addresses = len(addresses)

        if not self.deps_loaded:
            if not self.check_build():
                self.logger.info(f"Building dependencies for {self.type} parser")
                self.build_dependencies()
            self.logger.info(f"Loading dependencies for {self.type} parser")
            self.load_dependencies()
            self.deps_loaded = True

        self.logger.info(f"Beginning matching {n_addresses} addresses")

        if isinstance(addresses[0], ParsedAddress):
            addresses = cast(ParsedAddresses, addresses)
            return self._match_parsed_addresses(addresses, **kwargs)

        addresses = cast(Addresses, addresses)
        return self._match_unparsed_addresses(addresses, **kwargs)
