import logging
from pathlib import Path
from typing import final, override

from glam.logs import setup_logger

# matching
from glam.matching.base_matcher import BaseMatcher
from glam.matching.matched_address import MatchedAddress, MatchedAddresses
from glam.matching.matchers import get_matcher

# parsing
from glam.parsing.base_parser import BaseParser
from glam.parsing.parsed_address import ParsedAddresses
from glam.parsing.parsers import get_parser

# misc
from glam.preprocessing.preprocessor import PreProcessor
from glam.types import NZSA, Addresses, AddressIDs, Confidences
from glam.utils.utils import download_dependencies


@final
class Geocoder:
    """A class for interacting with geocoding methods.

    The Geocoder class provides functionality to parse and match addresses
    using configurable parsing and matching strategies.

    Args:
        data_dir (str): Path to data directory for glam dependencies
        matcher (str, optional): Matcher type to use. Defaults to "vector"
        parser (str, optional): Parser type to use. Defaults to "libpostal"
        log_level (int, optional): Logging level. Defaults to logging.INFO

    Attributes:
        parser (BaseParser | None): Parser instance used for address parsing
        matcher (BaseMatcher): Matcher instance used for address matching
        data_dir (Path): Path to data directory
        logger (Logger): Logger instance

    Raises:
        ValueError: If an incompatible parser-matcher combination is used

    Example:
        >>> gc = Geocoder(data_dir="/path/to/data")
        >>> results = gc.geocode_addresses(["123 Main St, City"])
    """

    parser: BaseParser | None = None
    matcher: BaseMatcher

    def __init__(
        self,
        data_dir: str,
        matcher: str = "tfidf",
        parser: str = "rnn",
        matcher_options: dict[str, object] | None = None,
        parser_options: dict[str, object] | None = None,
        nzsa_path: str | Path | None = None,
        paf_path: str | None = None,
        pnf_path: str | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        self.logger = setup_logger(log_level=log_level)
        self.data_dir = Path(data_dir)

        default_nzsa_path = self.data_dir / "nz-street-address.csv"
        if nzsa_path is None:
            nzsa_path = default_nzsa_path
            if not Path.is_file(nzsa_path):
                self.logger.info("Downloading glam dependencies")
                download_dependencies(self.data_dir)

        self.logger.info(f"Loading NZSA data from {nzsa_path}")
        NZSA.load(nzsa_path)

        # postcodes
        if NZSA.postcodes:
            self.logger.info("Using existing postcodes")
        elif pnf_path:
            self.logger.info(f"Integrating PNF data from {pnf_path}")
            NZSA.integrate_pnf(pnf_path, default_nzsa_path)
            self.logger.info(
                f"Saved updated address file with postcodes to {default_nzsa_path}"
            )
        elif paf_path:
            self.logger.info(f"Integrating PAF data from {paf_path}")
            NZSA.integrate_paf(paf_path, default_nzsa_path)
            self.logger.info(
                f"Saved updated address file with postcodes to {default_nzsa_path}"
            )

        else:
            self.logger.info(
                "No PNF or PAF data provided. Postcodes will be ignored when geocoding"
            )

        if matcher_options is None:
            matcher_options = dict()
        if parser_options is None:
            parser_options = dict()

        self.matcher = get_matcher(matcher)(self.data_dir, **matcher_options)
        self.parser = get_parser(parser)(self.data_dir, **parser_options)

        self._check_parser_matcher_combination()

    @override
    def __repr__(self) -> str:
        repr_str = (
            f"Geocoder\n Data directory: {self.data_dir}\n Matcher: {self.matcher.type}"
        )
        if self.parser:
            repr_str += f"\n Parser: {self.parser.type}"
        return repr_str

    def _set_parser(self, parser: str, **kwargs: object) -> BaseParser:
        """Set parser to be used when geocoding addresses."""
        return parsers[parser](self.data_dir, **kwargs)

    def _set_matcher(self, matcher: str, **kwargs: object) -> BaseMatcher:
        """Set matcher to be used when geocoding addresses."""
        return matchers[matcher](self.data_dir, **kwargs)

    def preprocess_addresses(self, addresses: Addresses) -> Addresses:
        """Preprocess addresses before parsing or matching.

        Args:
            addresses (Addresses): Input addresses to preprocess

        Returns:
            Addresses: Cleaned and standardized addresses
        """
        processor = PreProcessor()
        return processor.clean_addresses(addresses)

    def parse_addresses(
        self,
        addresses: Addresses,
        preprocess: bool = True,
        postprocess: bool = True,
    ) -> ParsedAddresses:
        """Parse addresses using the configured parser.

        Args:
            addresses (Addresses): Input addresses to parse
            preprocess (bool, optional): Whether to preprocess addresses. Default True
            postprocess (bool, optional): Whether to postprocess results. Default True

        Returns:
            ParsedAddresses: Parsed address components

        Raises:
            ValueError: If no parser is configured
        """
        if not self.parser:
            raise ValueError("Parser not set")

        if preprocess:
            addresses = self.preprocess_addresses(addresses)

        return self.parser.parse_addresses(addresses, post_process=postprocess)

    def match_addresses(
        self, addresses: Addresses | ParsedAddresses, **kwargs: object
    ) -> tuple[AddressIDs, Confidences]:
        """Match addresses against reference data using the configured matcher.

        Args:
            addresses (Addresses | ParsedAddresses): Input addresses to match

        Returns:
            MatchedAddresses: Matched address results

        Raises:
            ValueError: If no matcher is configured
        """
        if not self.matcher:
            raise ValueError("Matcher not set")

        return self.matcher.match_addresses(addresses, **kwargs)

    def geocode_addresses(
        self,
        addresses: Addresses,
        preprocess: bool = True,
        **kwargs: object,
    ) -> MatchedAddresses:
        """Geocode addresses by parsing and matching them.

        This is the main method that combines preprocessing, parsing (if required),
        and matching steps into a single operation.

        Args:
            addresses (Addresses): Input addresses to geocode
            preprocess (bool, optional): Whether to preprocess addresses. Default True

        Returns:
            MatchedAddresses: Geocoded address results
        """
        if self.matcher.requires_parser:
            parsed_addresses = self.parse_addresses(addresses, preprocess=preprocess)
            address_ids, confidences = self.match_addresses(parsed_addresses, **kwargs)
        else:
            address_ids, confidences = self.match_addresses(addresses, **kwargs)

        # match addresses to NZSA data
        linz_addresses = NZSA.get_addresses(address_ids)

        if self.matcher.requires_parser:
            fields = zip(addresses, parsed_addresses, linz_addresses, confidences)
            return [
                MatchedAddress(
                    search_address=i, parsed_address=j, matched_address=k, confidence=f
                )
                for i, j, k, f in fields
            ]

        fields = zip(addresses, linz_addresses, confidences)
        return [
            MatchedAddress(search_address=i, matched_address=j, confidence=f)
            for i, j, f in fields
        ]

    def _check_parser_matcher_combination(self) -> None:
        if self.matcher.requires_parser and not self.parser:
            raise ValueError(
                f"Matcher {self.matcher.type} requires a parser, but none was provided"
            )

        # if not self.matcher.requires_parser and self.parser:
        #     self.logger.warning(
        #         f"Parser {self.parser.type} is not required for matcher {self.matcher.type}"
        #     )


def data_dir_validation(data_dir: Path) -> bool:
    expected_linz_location = Path(data_dir) / "nz-street_address.csv"
    return Path.is_dir(data_dir) and Path.is_file(expected_linz_location)
