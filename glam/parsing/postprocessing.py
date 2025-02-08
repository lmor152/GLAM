import re

from glam.utils import lookups


def normalise_street_type(street_name: str) -> str:
    split_text = street_name.split()

    clean_suffix = re.sub(r"[ ,.\-]", "", split_text[-1])

    if clean_suffix in lookups.street_abbreviations:
        return " ".join(split_text[:-1] + [lookups.street_abbreviations[clean_suffix]])

    return street_name


def normalise_post_code(postcode: str) -> str | None:
    postcode = re.sub(r"[a-zA-Z,.\- ]", "", postcode)
    if len(postcode) != 4:
        return None
    return postcode


def normalise_unit(unit: str) -> str:
    numbers = re.sub(r"[a-zA-Z,.\- ]", "", unit)
    if len(numbers) > 0:
        return numbers

    for word in unit.lower().split():
        if word in lookups.ordinal_words:
            return str(lookups.ordinal_words.index(word) + 1)
        if word in lookups.cardinal_words:
            return str(lookups.cardinal_words.index(word) + 1)

    return unit


def normalise_level(level: str) -> str:
    numbers = re.sub(r"[a-zA-Z,.\- ]", "", level)
    if len(numbers) > 0:
        return numbers

    for word in level.lower().split():
        if word in lookups.ordinal_words:
            return str(lookups.ordinal_words.index(word) + 1)
        if word in lookups.cardinal_words:
            return str(lookups.cardinal_words.index(word) + 1)

    return level


def normalise_first_number(first_number: str) -> str | None:
    first_number = re.sub(r"[^0-9]", "", first_number)
    if len(first_number) == 0:
        return None
    return first_number


def remove_illegal_chars(string: str) -> str:
    return re.sub("[^A-Za-z0-9 ,]+", "", string)
