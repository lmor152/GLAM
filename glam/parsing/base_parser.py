from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import override

from glam.logs import setup_logger
from glam.parsing.parsed_address import ParsedAddresses
from glam.types import Addresses


class BaseParser(ABC):
    type: str
    logger: Logger = setup_logger()

    def __init__(self, data_dir: Path) -> None:
        self.data_dir: Path = data_dir
        self.loaded: bool = False

    @override
    def __repr__(self) -> str:
        return self.type

    @abstractmethod
    def _parse_addresses(
        self,
        addresses: Addresses,
    ) -> ParsedAddresses:
        pass

    @abstractmethod
    def load_dependencies(self) -> None:
        pass

    def post_process_addresses(self, addresses: ParsedAddresses) -> ParsedAddresses:
        for addy in addresses:
            addy.post_process()
        return addresses

    def parse_addresses(
        self,
        addresses: Addresses,
        post_process: bool = True,
        **kwargs: dict[str, object],
    ) -> ParsedAddresses:
        n_addresses = len(addresses)
        if not self.loaded:
            self.logger.info(f"Loading dependencies for {self.type} parser")
            self.load_dependencies()
            self.loaded = True
            self.logger.info(f"Dependencies loaded for {self.type} parser")

        self.logger.info(f"Beginning parsing {n_addresses} addresses")
        parsed_addresses = self._parse_addresses(addresses, **kwargs)
        if post_process:
            self.logger.info("Post processing parsed addresses")
            return self.post_process_addresses(parsed_addresses)

        return parsed_addresses
