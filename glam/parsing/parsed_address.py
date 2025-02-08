from collections.abc import Sequence
from typing import final, override

from glam.parsing import postprocessing
from glam.types import AddressComponent


def clean_component(component: AddressComponent) -> str | None:
    return str(component) if component else None


@final
class ParsedAddress:
    __slots__: list[str] = [
        "unit",
        "building",
        "level",
        "first_number",
        "first_number_suffix",
        "second_number",
        "street_name",
        "suburb_town_city",
        "postcode",
    ]

    def __init__(
        self,
        unit: AddressComponent = None,
        building: AddressComponent = None,
        level: AddressComponent = None,
        first_number: AddressComponent = None,
        first_number_suffix: AddressComponent = None,
        second_number: AddressComponent = None,
        street_name: AddressComponent = None,
        suburb_town_city: AddressComponent = None,
        postcode: AddressComponent = None,
    ) -> None:
        self.unit = clean_component(unit)
        self.building = clean_component(building)
        self.level = clean_component(level)
        self.first_number = clean_component(first_number)
        self.first_number_suffix = clean_component(first_number_suffix)
        self.second_number = clean_component(second_number)
        self.street_name = clean_component(street_name)
        self.suburb_town_city = clean_component(suburb_town_city)
        self.postcode = clean_component(postcode)

    @override
    def __repr__(self) -> str:
        return f"ParsedAddress{str(self.to_dict())}"

    @override
    def __str__(self) -> str:
        return self.format_address()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParsedAddress):
            return False
        return self.to_dict() == other.to_dict()

    def has(self, attr: str) -> bool:
        return self.__getattribute__(attr) is not None

    def get(self, attr: str, default: str = "") -> str:
        got = getattr(self, attr)
        if got is None:
            return default
        return got

    def to_dict(self) -> dict[str, str | None]:
        return {slot: getattr(self, slot) for slot in self.__slots__}

    @staticmethod
    def join_parts(*parts: object, sep: str = "") -> str:
        return sep.join([x for x in parts if x is not None and len(x) > 0])

    def format_address(self, human: bool = False) -> str:
        sep = " " if human else "|"

        head = ""
        if self.unit:
            head += self.unit
        if self.unit and self.first_number and human:
            head += "/"

        if self.first_number:
            head = self.join_parts(head, self.first_number, sep=sep)

        if self.first_number and self.first_number_suffix:
            head = self.join_parts(head, self.first_number_suffix, sep=sep)

        if self.second_number:
            if len(head) > 0:
                head += "-" if human else "|" + self.second_number
            else:
                head = self.join_parts(head, self.second_number, sep=sep)

        middle_parts = [self.street_name, self.suburb_town_city]
        middle_parts = [x for x in middle_parts if x]  # take out None values
        middle_parts = self.join_parts(*middle_parts, sep=sep)

        addy_parts = [head, middle_parts, self.postcode]

        return self.join_parts(*addy_parts, sep=sep)

    def post_process(self) -> None:
        self.street_name = (
            postprocessing.normalise_street_type(self.street_name)
            if self.street_name
            else None
        )
        self.first_number = (
            postprocessing.normalise_first_number(self.first_number)
            if self.first_number
            else None
        )
        self.unit = postprocessing.normalise_unit(self.unit) if self.unit else None
        self.postcode = (
            postprocessing.normalise_post_code(self.postcode) if self.postcode else None
        )
        self.level = postprocessing.normalise_level(self.level) if self.level else None  # noqa: E501


ParsedAddresses = Sequence[ParsedAddress]
