from pathlib import Path
from typing import final, override
import platform

import torch

from glam.logs import get_logger
from glam.parsing.base_parser import BaseParser
from glam.parsing.parsed_address import ParsedAddresses
from glam.parsing.rnn import inference
from glam.parsing.rnn.model import AddressParser, address_parser_model
from glam.types import Addresses


@final
class RNNParser(BaseParser):
    model: AddressParser  
    device: torch.device 
    type: str = "RNN"

    @override
    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir)

    @override
    def load_dependencies(self) -> None:
        self.device = self.get_device()
        model_path = Path(__file__).parent / "model.pth"
        self.model = self.load_model(model_path)

    @override
    def _parse_addresses(
        self, addresses: Addresses, batch_size: int = 256
    ) -> ParsedAddresses:
        return inference.parse_addresses(addresses, self.model, batch_size, self.device)

    def get_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        logger = get_logger()
        logger.info(f"Using device {device}")
        return device

    def load_model(self, model_path: Path) -> AddressParser:
        _ = address_parser_model.to(self.device)
        _ = address_parser_model.load_state_dict(
            torch.load(model_path, weights_only=True) 
        )

        return address_parser_model
