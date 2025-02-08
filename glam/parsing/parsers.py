from typing import Type

from glam.parsing.base_parser import BaseParser
from glam.parsing.libpostal.postal_parser import PostalParser
from glam.parsing.rnn.rnn_parser import RNNParser

parsers: dict[str, Type[BaseParser]] = {"rnn": RNNParser, "libpostal": PostalParser}
