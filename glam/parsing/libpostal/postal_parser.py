import re
from typing import final, override

from postal import parser as libpostal

from glam.parsing.base_parser import BaseParser
from glam.parsing.parsed_address import ParsedAddress, ParsedAddresses
from glam.types import Addresses

LibpostalParsed = list[tuple[str, str]]


@final
class PostalParser(BaseParser):
    type = "LibPostal"

    @override
    def _parse_addresses(self, addresses: Addresses) -> ParsedAddresses:
        return [glam_parser(address) for address in addresses]

    @override
    def load_dependencies(self) -> None:
        pass  # no dependencies required


def glam_parser(address: str) -> ParsedAddress:
    raw_components: LibpostalParsed = libpostal.parse_address(
        address, country="nz", language="en"
    )

    component_map = {
        "house": "building",
        # 'category':       None,
        # 'near':           None,
        "house_number": "house_number",  # requires additional conversion
        "road": "street_name",
        # 'unit':           None,
        "level": "level",
        # 'staircase':      None,
        # 'entrance':       None,
        # 'po_box':         None,
        "postcode": "postcode",
        "suburb": "suburb_town_city",
        "city_district": "suburb_town_city",
        "city": "suburb_town_city",
        "island": "suburb_town_city",
        # 'state_district': None,
        # 'state':          None,
        # 'country_region': None,
        # 'country':        None,
        # 'world_region':   None,
    }

    # build the address dictionary
    address_dict: dict[str, str] = {}
    for val, key in raw_components:
        if key in component_map:
            glam_key = component_map[key]
            if glam_key in address_dict:
                address_dict[glam_key] += " " + val.upper()
            else:
                address_dict[glam_key] = val.upper()

    # convert house number
    if "house_number" in address_dict:
        fixed_numbers = convert_house_number(address_dict["house_number"])
        address_dict.update(fixed_numbers)
        del address_dict["house_number"]

    return ParsedAddress(**address_dict)


def convert_house_number(house_number: str) -> dict[str, str]:
    # Extract all digits
    digits = re.findall(r"\d+", house_number)

    # Extract all non-digits
    non_digits = re.findall(r"[^,\d;|/\-\s]+", house_number)

    output = {}
    if len(digits) == 1:
        output["first_number"] = digits[0]
    elif len(digits) > 1:
        output["unit"] = digits[0]
        output["first_number"] = digits[1]

    for nd in non_digits:
        if len(nd) == 1:
            output["first_number_suffix"] = nd

    return output
