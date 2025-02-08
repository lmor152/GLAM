import math
import re
from re import Pattern

import numpy as np

from glam.parsing.parsed_address import ParsedAddress

# create letter groups that are the alphabet minus one letter
alphabet = " abcdefghijklmnopqrstuvwxyz"
letter_groups = list(alphabet)
letter_groups = [re.sub(f"[{lg}]", "", alphabet) for lg in letter_groups]  # inverted
letter_groups_compiled = [re.compile(f"[{lg}]", flags=re.I) for lg in letter_groups]


def idx_geom_mean(street: str, lg: str) -> float:
    l = [f.span()[0] + 1 for f in re.finditer(f"[{lg}]", street, flags=re.I)]
    b = len(l)

    # very fast way to calculate geometric mean
    return b and math.exp(sum(map(math.log, l)) / b) or 1


def idx_geom_mean_compiled(street: str, lg: Pattern[str]) -> float:
    l = [f.span()[0] + 1 for f in lg.finditer(street)]
    b = len(l)

    # very fast way to calculate geometric mean
    return b and math.exp(sum(map(math.log, l)) / b) or 1


def gm_vectorise(text: str, scale: float, lgs: list[Pattern[str]]) -> list[float]:
    nums = re.sub(r"[^\d+]", "", text)
    chars = re.sub(r"[\d+]", "", text)
    length = max(len(chars), 1)  # to avoid problems with empty strings

    embeddings = [idx_geom_mean_compiled(chars, lg) * scale / length for lg in lgs]
    nums = -1 if len(nums) == 0 else np.log(int(nums) + 1)

    return embeddings + [np.log(length)] + [nums]


def divide(a: float, b: float) -> float:
    return b and a / b or 0


weights: dict[str, float] = {
    "unit": 0.1,
    "first_number": 5,
    "first_number_suffix": 1,
    "street_name": 100,
    "suburb_town_city": 5,
    "postcode": 10,
}


class VectorisingFuncs:
    @staticmethod
    def unit(x: str | None) -> list[float]:
        if x is None:
            return [weights["unit"] * -1]

        digits = re.sub(r"[^\d+]", "", x)  # keep only digits
        return (
            [weights["unit"] * int(digits) + 1]
            if len(digits)
            else [weights["unit"] * -1]
        )

    @staticmethod
    def first_number(x: str) -> list[float]:
        digits = re.sub(r"[^\d+]", "", x)
        if len(digits) == 0:
            return [weights["first_number"] * -1]
        return [weights["first_number"] * np.log(int(x) + 10)]

    @staticmethod
    def first_number_suffix(x: str | None) -> list[float]:
        if x is None:
            return [weights["first_number_suffix"] * -1]
        return [
            weights["first_number_suffix"]
            * (sum(map(ord, x)) + 1 - len(x) * (ord("A")) / (ord("Z") - ord("A")))
        ]

    @staticmethod
    def street_name(x: str) -> list[float]:
        return gm_vectorise(x, weights["street_name"], letter_groups_compiled)

    @staticmethod
    def suburb_town_city(x: str) -> list[float]:
        #remove duplicate words
        words = re.findall(r'\b\w+\b', x, flags=re.IGNORECASE)
    
        stc = " ".join(set(words))
        return gm_vectorise(stc, weights["suburb_town_city"], letter_groups_compiled)

    @staticmethod
    def postcode(x: str | None) -> list[float]:
        if x is None:
            return [weights["postcode"] * -1]
        return [weights["postcode"] * int(x) / 9999]


def make_vectors(
    parsed_address: ParsedAddress | None,
    postcode: bool = True,
) -> list[float]:
    vector: list[float] = []

    # exit with empty vector if street number and name not present
    if parsed_address is None:
        return []
    if not parsed_address.has("first_number") and not parsed_address.has("street_name"):
        return []

    # ensure only digits from unit are used
    vector += VectorisingFuncs.unit(parsed_address.get("unit"))

    vector += VectorisingFuncs.first_number(parsed_address.get("first_number"))

    vector += VectorisingFuncs.first_number_suffix(
        parsed_address.get("first_number_suffix")
    )

    vector += VectorisingFuncs.street_name(parsed_address.get("street_name"))

    if parsed_address.has("suburb_town_city"):
        vector += VectorisingFuncs.suburb_town_city(
            parsed_address.get("suburb_town_city")
        )

    if postcode and parsed_address.has("postcode"):
        vector += VectorisingFuncs.postcode(parsed_address.get("postcode"))

    return vector
