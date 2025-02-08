from typing import final

from glam.preprocessing.cleaners import all_cleaners
from glam.types import Addresses


@final
class PreProcessor:
    def __init__(self) -> None:
        self.cleaning_funcs = all_cleaners

    def clean_addresses(self, addresses: Addresses) -> Addresses:
        """Cleans addresses"""

        addresses = list(addresses)

        for fh in self.cleaning_funcs:
            addresses = fh(addresses)

        return addresses
