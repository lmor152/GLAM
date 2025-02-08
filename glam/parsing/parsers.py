import importlib
from typing import Type

from glam.parsing.base_parser import BaseParser

parsers: dict[str, str] = {
    "rnn": "glam.parsing.rnn.rnn_parser.RNNParser",
    "libpostal": "glam.parsing.libpostal.postal_parser.PostalParser",
}


def get_parser(parser_name: str) -> Type[BaseParser]:
    parser_path = parsers.get(parser_name)
    if parser_path is None:
        raise ValueError(f"Unknown parser: {parser_name}")

    # Dynamically import the parser module
    module_name, class_name = parser_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
