import pickle
from pathlib import Path
from typing import Callable

import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from glam import logs
from glam.matching.vector.model import VectorisingFuncs
from glam.types import NZSA

# tqdm.pandas()

logger = logs.get_logger()

# # parsed : nzsa
# argmap = {
#     "unit": "unit_value",
#     "first_number": "address_number",
#     "first_number_suffix": "address_number_suffix",
#     "street_name": "full_road_name_ascii",
#     "suburb_town_city": "suburb_town_city",
#     "postcode": "POSTCODE",
# }


# def parse_linz_address(
#     row: dict[str, AddressComponent], comb: list[str]
# ) -> ParsedAddress:
#     return ParsedAddress(**{k: row[argmap[k]] for k in comb})


# def build_tree(comb: list[str]) -> cKDTree:
#     # need to fill na for columns in the combination
#     # None values would result in inhomogeneous vector length
#     temp_df = NZSA.df[[argmap[x] for x in comb]].fillna(" ")  # type: ignore

#     # create parsed addresses from the NZSA data
#     linz_addresses: ParsedAddresses = list(
#         temp_df.apply(lambda x: parse_linz_address(x, comb), axis=1)  # type: ignore
#     )

#     # convert parsed addresses to vectors
#     vectors = [make_vectors(addy) for addy in tqdm(linz_addresses)]

#     return cKDTree(np.array(vectors))


# def build_vectors(out_dir: Path) -> None:
#     if NZSA.df is None:
#         raise ValueError("NZSA data not loaded")

#     # for reduced trees, base values are always present
#     base = ["unit", "first_number", "first_number_suffix", "street_name"]

#     vector_combinations = [
#         base,
#         base + ["suburb_town_city"],
#     ]

#     if NZSA.postcodes:
#         vector_combinations += [vc + ["postcode"] for vc in vector_combinations]

#     for i, comb in enumerate(vector_combinations):
#         logger.info(f"Building tree {i+1}/{len(vector_combinations)}...")
#         tree = build_tree(comb)

#         # save tree
#         save_loc = out_dir / f"tree{i}.pkl"
#         with Path.open(save_loc, "wb") as f:
#             pickle.dump(tree, f)

#         idx_map = zip(range(len(NZSA.df)), NZSA.df["address_id"], strict=True)  # type: ignore
#         pd.DataFrame(idx_map, columns=["idx", "address_id"]).to_csv(
#             out_dir / "idx_map.csv", index=False
#         )

#         # flag for if postcodes were used
#         if NZSA.postcodes:
#             with Path.open(out_dir / "postcodes.txt", "w") as f:
#                 f.write(str(NZSA.postcodes))


class VectorInstruction:
    def __init__(
        self,
        name: str,
        field: str,
        func: Callable[[str | None], list[float]] | Callable[[str], list[float]],
        fillna: str | int | None = None,
    ) -> None:
        self.name = name
        self.field = field
        self.func = func
        self.fillna = fillna


def build_vectors2(out_dir: Path) -> None:
    if NZSA.df is None:
        raise ValueError("NZSA data not loaded")

    # tree combinations
    combinations: list[tuple[str, ...]] = [
        (
            "unit_vector",
            "address_number_vector",
            "address_number_suffix_vector",
            "street_name_vector",
        ),
        (
            "unit_vector",
            "address_number_vector",
            "address_number_suffix_vector",
            "street_name_vector",
            "suburb_town_city_vector",
        ),
    ]

    vectors_df = NZSA.df.copy()

    to_vectorise = [
        VectorInstruction(
            name="unit_vector", field="unit_value", func=VectorisingFuncs.unit
        ),
        VectorInstruction(
            name="address_number_vector",
            field="address_number",
            func=VectorisingFuncs.first_number,
        ),
        VectorInstruction(
            name="address_number_suffix_vector",
            field="address_number_suffix",
            func=VectorisingFuncs.first_number_suffix,
        ),
        VectorInstruction(
            name="street_name_vector",
            field="full_road_name_ascii",
            func=VectorisingFuncs.street_name,
        ),
        VectorInstruction(
            name="suburb_town_city_vector",
            field="suburb_town_city",
            func=VectorisingFuncs.suburb_town_city,
            fillna="",
        ),
    ]

    if NZSA.postcodes:
        to_vectorise.append(
            VectorInstruction(
                name="postcode_vector", field="postcode", func=VectorisingFuncs.postcode
            )
        )
        combinations += [c + ("postcode_vector",) for c in combinations]

    for i, instr in enumerate(to_vectorise):
        logger.info(f"Vectorising field {i+1}/{len(to_vectorise)} ({instr.field})")
        if instr.fillna is not None:
            vectors_df[instr.field] = vectors_df[instr.field].fillna(instr.fillna)
        vectors_df[instr.name] = [instr.func(x) for x in tqdm(vectors_df[instr.field])]

    for i, comb in enumerate(combinations):
        logger.info(f"Building tree {i+1}/{len(combinations)}")
        vectors = get_vectors(vectors_df, comb)
        tree = cKDTree(vectors)

        # save tree
        save_loc = out_dir / f"tree{i}.pkl"
        with Path.open(save_loc, "wb") as f:
            pickle.dump(tree, f)

    idx_map = zip(range(len(NZSA.df)), NZSA.df["address_id"], strict=True)
    pd.DataFrame(idx_map, columns=["idx", "address_id"]).to_csv(
        out_dir / "idx_map.csv", index=False
    )

    # flag for if postcodes were used
    if NZSA.postcodes:
        with Path.open(out_dir / "postcodes.txt", "w") as f:
            f.write(str(NZSA.postcodes))


def get_vectors(vector_df: pd.DataFrame, comb: tuple[str, ...]) -> list[list[float]]:
    def extend(a: list[list[float]]) -> list[float]:
        out = []
        for sublist in a:
            out.extend(sublist)
        return out

    array = vector_df[list(comb)].to_numpy()
    vectors = []
    for row in array:
        vectors.append(extend(row))
    return vectors
