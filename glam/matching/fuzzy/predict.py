import gc
from typing import Callable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm

from glam.parsing.parsed_address import ParsedAddress, ParsedAddresses
from glam.types import AddressID, Confidence

weights = {
    "unit_value_score": 1,
    "address_number_score": 20,
    "address_number_suffix_score": 1,
    "address_number_high_score": 1,
    "full_road_name_ascii_score": 100,
    "suburb_town_city_score": 50,
    "postcode_score": 20,
}


def lookup_addresses(
    addresses: ParsedAddresses,
    nzsa: pd.DataFrame,
    speed_performance_balance: int = 30,
) -> tuple[list[AddressID | None], list[Confidence]]:
    matched_addresses: list[tuple[AddressID, Confidence] | None] = []
    for i, x in enumerate(
        tqdm(addresses, smoothing=0.05, mininterval=1, colour="green")
    ):
        matched_addresses.append(
            lookup_address_rapidfuzz(x, nzsa, weights, speed_performance_balance)
        )
        if i % 1000 == 0:
            gc.collect()
    gc.collect()

    return list(zip(*matched_addresses))  # type: ignore


def lookup_address_rapidfuzz(
    addy: ParsedAddress,
    nzsa: pd.DataFrame,
    weights: dict[str, int] = weights,
    confidence: int = 50,
) -> tuple[AddressID | None, Confidence] | None:
    """
    Takes a parsed address and finds the best match in the linz dataset

    Inputs:
        addy: search address in parsed dictionary format
        nzsa: pandas df of linz addresses
    Outputs:
        dictionary of address components with latitute and longitude
    """

    if addy.first_number is None and addy.street_name is None:
        return None, 0

    search_area = nzsa
    using_postcodes = "postcode_int" in search_area.columns

    # step 1: exact matching
    # step 1.1: exact match on postcode if available
    if addy.has("postcode") and using_postcodes:
        search_area = reduce_search_space(
            "postcode_int",
            int(addy.postcode),  # type: ignore
            search_area=search_area,
            matcher="exact",
        )

    # step 1.2: exact match on street number if available
    if addy.has("first_number"):
        search_area = reduce_search_space(
            "address_number_int",
            int(addy.first_number),  # type: ignore
            search_area=search_area,
            matcher="exact",
        )

    # step 1.3: revert if no exact matches
    # if exact matching failed, try again with fuzzy matching
    if len(search_area) == 0:
        search_area = nzsa

        if addy.has("first_number"):
            search_area["address_number_score"] = 10000 / (
                abs(search_area["address_number_int"] - int(addy.first_number)) + 100
            )

        if addy.has("postcode") and using_postcodes:
            search_area = reduce_search_space(
                "postcode",
                addy.postcode,  # type: ignore
                search_area=search_area,
            )

    # step 2: fuzzy matching
    if addy.has("street_name"):
        search_area = reduce_search_space(
            "full_road_name_ascii",
            addy.street_name,  # type: ignore
            search_area=search_area,
            matcher=fuzz.ratio,
            confidence=95,  # street names are important
        )

    if len(search_area) == 0:
        # this means the address is probably not in NZSA
        # retry with fuzzy
        search_area = nzsa

        if addy.has("first_number"):
            search_area["address_number_score"] = 10000 / (
                abs(search_area["address_number_int"] - int(addy.first_number)) + 100
            )
            search_area = search_area[search_area["address_number_score"] > 85]

        if addy.has("postcode") and using_postcodes:
            search_area = reduce_search_space(
                "postcode",
                addy.postcode,  # type: ignore
                search_area=search_area,
                confidence=50,
            )

        if addy.has("street_name"):
            search_area = reduce_search_space(
                "full_road_name_ascii",
                addy.street_name,  # type: ignore
                search_area=search_area,
                matcher=fuzz.ratio,
                confidence=70,  # street names are important
            )

    if addy.has("suburb_town_city"):
        search_area = reduce_search_space(
            "suburb_town_city",
            addy.suburb_town_city,  # type: ignore
            search_area=search_area,
            matcher=fuzz.partial_ratio,
            confidence=confidence,
        )

    if addy.has("unit"):
        search_area = reduce_search_space(
            "unit_value",
            addy.unit,  # type: ignore
            search_area=search_area,
            matcher=fuzz.ratio,
            confidence=0,
        )

    if addy.has("first_number_suffix"):
        search_area = reduce_search_space(
            "address_number_suffix",
            addy.first_number_suffix,  # type: ignore
            search_area=search_area,
            matcher=fuzz.ratio,
            confidence=0,
        )

    return conclude_search(search_area, weights)


def conclude_search(
    search_area: pd.DataFrame, weights: dict[str, int]
) -> tuple[AddressID | None, Confidence] | None:
    if len(search_area) == 0:
        return None, 0

    # only use columns that mattered
    score_cols = [col for col in weights if col in search_area.columns]

    search_area["match_score"] = np.sum(
        [weights[col] * search_area[col] for col in score_cols], axis=0
    ) / sum([weights[col] for col in score_cols])

    match = search_area.loc[search_area["match_score"].idxmax()]
    return match["address_id"], match["match_score"]


def reduce_search_space(
    search_col: str,
    search_term: str | int,
    search_area: pd.DataFrame,
    matcher: Callable[..., float] | str = fuzz.ratio,
    confidence: int = 20,
) -> pd.DataFrame:
    """
    helper function for address lookup to iteratively
    reduce search space in LINZ dataset
    """

    # perform an exact match
    if matcher == "exact":
        search_area = search_area[search_area[search_col].values == search_term]

    # uses the provided matching function
    else:
        res = process.extract(  # type: ignore
            search_term,  # type: ignore
            search_area[search_col].str.upper(),  # type: ignore
            # scorer=matcher,  # type: ignore
            score_cutoff=confidence,
            limit=len(search_area),
        )

        search_area = search_area.loc[[x[2] for x in res]]
        search_area[search_col + "_score"] = [x[1] for x in res]

    return search_area


# def get_matches_df(sparse_matrix, A, B, top=100):
#     non_zeros = sparse_matrix.nonzero()

#     sparserows = non_zeros[0]
#     sparsecols = non_zeros[1]

#     nr_matches = top if top else sparsecols.size

#     left_side = np.empty([nr_matches], dtype=object)
#     right_side = np.empty([nr_matches], dtype=object)
#     similairity = np.zeros(nr_matches)

#     for index in range(0, nr_matches):
#         left_side[index] = A[sparserows[index]]
#         right_side[index] = B[sparsecols[index]]
#         similairity[index] = sparse_matrix.data[index]

#     return pd.DataFrame(
#         {"left_side": left_side, "right_side": right_side, "similairity": similairity}
#     )


# def join_address(addy):
#     parts = ""

#     if "unit" in addy:
#         parts += addy["unit"] + "/"

#     parts += addy.get("first_number", "")
#     parts += addy.get("first_number_suffix", "")

#     parts += " " + addy.get("street_name", "")

#     if "suburb_town_city" in addy:
#         parts += ", " + addy.get("suburb_town_city", "")

#     if "postcode" in addy:
#         parts += ", " + addy.get("postcode", "")

#     return parts
