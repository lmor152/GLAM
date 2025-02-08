import random
import re
import string

import numpy as np
import pandas as pd

from glam.utils import lookups, typo
from glam.utils.utils import choose

labels_list = [
    "blank",
    "building",
    "level",
    "unit",
    "first_number",
    "first_number_suffix",
    "second_number",
    "street_name",
    "suburb_town_city",
    "postcode",
]

# Number of labels in total
n_labels = len(labels_list)

# Allowable characters for the encoded representation
vocab = list(
    string.digits + string.ascii_lowercase + string.punctuation + string.whitespace
)


def vocab_lookup(characters):
    """
    Takes in a string and returns an list of encoded vocab indices
    Inputs:
        characters:        the character string to be encoded
    Outputs:
        return:            list of integer vocab indices
    """
    result = list()
    for c in characters.lower():
        try:
            result.append(vocab.index(c) + 1)
        except ValueError:
            pass
            # result.append(len(vocab)+1)
    return np.array(result)


def labels(text, field_name, mutate=True):
    """
    Takes a string and label class and creates label list. Typos are applied if mutate == True
    Inputs:
        text: the text to label
        field_name: name of the class to use as label
        mutate: whether to apply typos or not. Default is to apply typos
    Outputs:
        text and list of labels
    """

    # Ensure the input is a string, encoding None to an empty to string
    if text is None:
        text = ""
    else:
        # Introduce artificial typos if mutate == True
        text = typo.generate_typo(str(text)) if mutate else str(text)
    encoded = [labels_list.index(field_name)] * len(text)

    return text, encoded


def join_labels(lbls, sep=" "):
    """
    Concatenates a series of label matrices with a separator
    :param lbls: a list of labels matrices
    :param sep: the separator string or function that returns the sep string
    :return: the concatenated labels
    """
    if len(lbls) < 2:
        return lbls

    joined_labels = None
    sep_str = None

    # if `sep` is not a function, set the separator (`sep_str`) to `sep`, otherwise leave as None
    if not callable(sep):
        sep_str = sep

    for l in lbls:
        if joined_labels is None:
            joined_labels = l
        else:
            # If `sep` is a function, call it on each iteration
            if callable(sep):
                sep_str = sep()
            # Skip zero-length labels
            if len(l) == 0:
                continue
            if sep_str is not None and len(sep_str) > 0 and len(joined_labels) > 0:
                # Join using sep_str if it's present and non-zero in length
                joined_labels = np.concatenate(
                    [joined_labels, labels(sep_str, "blank", mutate=False)[1], l],
                    axis=0,
                )
            else:
                # Otherwise, directly concatenate the labels
                joined_labels = np.concatenate([joined_labels, l], axis=0)

    return joined_labels


def join_str_and_labels(parts, sep=" "):
    """
    Joins the strings and labels using the given separator
    :param parts: a list of string/label tuples
    :param sep: a string or function that returns the string to be used as a separator
    :return: the joined string and labels
    """

    # Keep only the parts with strings of length > 0
    parts = [p for p in parts if len(p[0]) > 0]

    # If there are no parts at all, return an empty string an array of shape (0, n_labels)
    if len(parts) == 0:
        return "", np.zeros((0, n_labels))
    # If there's only one part, just give it back as-is
    if len(parts) == 1:
        return parts[0]

    # Pre-generate the separators - this is important if `sep` is a function returning non-deterministic results
    n_sep = len(parts) - 1
    if callable(sep):
        seps = [sep() for _ in range(n_sep)]
    else:
        seps = [sep] * n_sep
    seps += [""]

    # Join the strings using the list of separators
    strings = "".join(sum([(s[0][0], s[1]) for s in zip(parts, seps)], ()))

    # Join the labels using an iterator function
    sep_iter = iter(seps)
    lbls = join_labels([s[1] for s in parts], sep=lambda: next(sep_iter))

    assert len(strings) == lbls.shape[0], (
        "string length %i (%s), label length %i using sep %s"
        % (len(strings), strings, lbls.shape[0], seps)
    )
    return strings, lbls


def generate_building_name(s):
    # do some random stuff and return as label

    if len(s) == 0:
        return labels(s, "building_name")

    def add_building(s):
        if random.getrandbits(1):
            s += " " + random.choice(lookups.dwelling_types)
        else:
            s += random.choice(lookups.dwelling_types) + " "
        return s

    s = choose(lambda: add_building(s), lambda: s)

    return labels(s, "building_name")


def generate_level(s):
    if len(s) == 0:
        return labels(s, "level")

    def number_first(s):
        if s == "0":
            return random.choice(lookups.level_types)
        r = random.choice(["ordinal", "ordinal_words"])
        return (
            lookups.num2word(s, r)
            + " "
            + random.choice(["level", "lvl", "floor", "flr"])
        )

    def number_last(s):
        if s == "0":
            return (
                random.choice(["floor", "level", "lv", "flr", "floor"])
                + " "
                + random.choice(["0", "zero"])
            )
        return (
            random.choice(["level", "lvl", "lv", "floor", "flr"])
            + " "
            + random.choice([lookups.num2word(s, "cardinal"), s])
        )

    s = choose(lambda: number_first(s), lambda: number_last(s))

    return labels(s, "level")


def generate_unit(s):
    if len(s) == 0:
        return labels(s, "unit")

    def add_dwelling_type(s):
        return random.choice(lookups.dwelling_types) + " " + s

    s = choose(lambda: add_dwelling_type(s), lambda: s, chance1=0.5)

    return labels(s, "unit")


def generate_first_number(s):
    return labels(s, "first_number")


def generate_first_number_suffix(s):
    return labels(s, "first_number_suffix")


def generate_second_number(s):
    return labels(s, "second_number")


def generate_street_name(s):
    if len(s) == 0:
        return labels(s, "street_name")

    # abbreviate street suffix
    def abbreviate_street(s):
        for k in lookups.street_abbreviations_reversed.keys():
            if k.lower() in s.lower():
                s = re.sub(
                    k + "$",
                    lookups.street_abbreviations_reversed[k],
                    s,
                    flags=re.IGNORECASE,
                )
                continue
        return s

    s = choose(lambda: abbreviate_street(s), lambda: s)

    return labels(s, "street_name")


def generate_suburb_town_city(s):
    def remove_vowels(s):
        return re.sub(r"[aeiou]", "", s, flags=re.IGNORECASE)

    words = s.split(" ")

    new_word = []
    for word in words:
        new_word.append(choose(lambda: remove_vowels(word), lambda: word, chance1=0.05))

    words = " ".join(new_word)

    if np.random.uniform() < 0.6:
        words.replace(",", "")

    return labels(words, "suburb_town_city")


def generate_postcode(s):
    return labels(s, "postcode")


def random_separator(min_length=1, max_length=3, possible_sep_chars=r",.-/\  "):
    """
    Generates a space-padded separator of random length using a random character from possible_sep_chars
    :param min_length: minimum length of the separator
    :param max_length: maximum length of the separator
    :param possible_sep_chars: string of possible characters to use for the separator
    :return: the separator string
    """
    chars = [" "] * random.randint(min_length, max_length)
    if len(chars) > 0 and possible_sep_chars:
        sep_char = random.choice(possible_sep_chars)
        chars[random.randrange(len(chars))] = sep_char
    return "".join(chars)


def sythnesise_address(record):
    """
    creates randomish address string with labels based on a linz record encoded from vocab and labels list

    Inputs:
        record: a row from the linz dataset
    Outputs:
        encoded address and labels
    """

    parts = []
    to_shuffle = []

    # gather all the parts
    building = generate_building_name(record["building_name"])
    level = generate_level(record["level_number"])
    unit = generate_unit(record["unit"])
    first_number = generate_first_number(record["address_number"])
    first_number_suffix = generate_first_number_suffix(record["first_number_suffix"])
    second_number = generate_second_number(record["second_number"])
    street = generate_street_name(record["full_road_name_ascii"])
    suburb_town_city = generate_suburb_town_city(record["suburb_town_city"])
    postcode = generate_postcode(record["postcode"])

    # if unit is just a number, combine it with the first number (with a separator), otherwise include it with building name and level
    if len(unit[0]) < 3:
        first_number = join_str_and_labels(
            [unit, first_number], sep=lambda: random_separator(1, 2, r",._/\ ")
        )
        head = [building, level]
    else:
        head = [building, level, unit]

    # include head in random order
    random.shuffle(head)
    parts += head

    # control the house number separators
    if (
        len(second_number[0]) > 0
    ):  # if there is a second number, don't use - to separate suffix
        seps = r"./\ "
    else:
        seps = r".-/\ "

    first_numbers = join_str_and_labels(
        [first_number, first_number_suffix], sep=lambda: random_separator(0, 1, seps)
    )

    street_numbers = join_str_and_labels([first_numbers, second_number], sep="-")
    # usually include the street numbers
    if np.random.uniform() <= 0.70:
        parts.append(street_numbers)

    # always include the street
    parts.append(street)

    tail = [postcode]
    choose(lambda: tail.append(suburb_town_city), chance1=0.5)
    random.shuffle(tail)
    parts += tail

    address, labels_encoded = join_str_and_labels(
        parts, sep=lambda: random_separator(1, 2, r",  ")
    )
    address_encoded = vocab_lookup(address)

    # for tf2
    return address_encoded, labels_encoded


def build_dataset(balanced_linz, outdir, n_records=0, val_split=0.2):
    """
    Generates training and validation tfrecord files from balanced linz csv which must be created beforehand
    Inputs:
        n_records:         the number of total (training + validation) rows to create from the balanced linz dataset. defaults to length of linz dataset
        val_split:         the portion of training data to withhold for validation set
        path:              the path to store the produced training and validation sets
    Outputs:
        return:            returns None. Saves training and validation tfrecord files to specified path
    """

    # load in the balanced linz dataset
    linz = pd.read_csv(balanced_linz, dtype=str).fillna("")

    # sample from linz dataset
    if n_records == 0:
        samp = linz
    else:
        samp = linz.sample(n_records)

    # apply randomness and encoding addresses + labels
    print("  sythnesising addresses")
    samp["train"] = samp.apply(sythnesise_address, axis=1)

    # apply test/val split
    val = samp.sample(frac=val_split)
    train = samp.drop(val.index)

    return train, val
