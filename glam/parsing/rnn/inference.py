import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from glam.parsing.parsed_address import ParsedAddress, ParsedAddresses
from glam.parsing.rnn.dataset import labels_list, vocab
from glam.parsing.rnn.model import AddressParser
from glam.types import Address, Addresses

char_to_index = {char: i + 1 for (i, char) in enumerate(vocab)}


def parse_addresses(
    addresses: Addresses, model: AddressParser, batch_size: int, device: torch.device
) -> ParsedAddresses:
    encoded = [[char_to_index.get(char, 0) for char in addy] for addy in addresses]

    preds = model_inference(encoded, model, batch_size, device)
    return [make_mappings(addy, res) for addy, res in zip(addresses, preds)]


def model_inference(
    encoded_addresses: list[list[int]],
    model: AddressParser,
    batch_size: int,
    device: torch.device,
) -> list[list[int]]:
    def inference_padder(batch: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        padded_tensors = pad_sequence(
            [torch.tensor(x) for x in batch], batch_first=True, padding_value=0
        )
        mask = padded_tensors != 0

        return padded_tensors, mask

    data_loader = DataLoader(
        encoded_addresses,  
        batch_size=batch_size,
        shuffle=False,
        collate_fn=inference_padder,
    )

    all_predictions = []
    model = model.to(device)
    _ = model.eval()
    with torch.no_grad():
        for batch, mask in data_loader:
            batch = batch.to(device)
            mask = mask.to(device)
            output = model(batch, mask)
            preds = torch.argmax(output, dim=-1).cpu().numpy()
            all_predictions.extend(preds)

    return all_predictions


def make_mappings(addy: Address, res: list[int]) -> ParsedAddress:
    mappings: dict[str, str] = dict()
    for char, class_id in zip(addy.upper(), res):
        if class_id == 0:
            continue
        cls = labels_list[class_id]
        mappings[cls] = mappings.get(cls, "") + char

    return ParsedAddress(**mappings)
