import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from glam.matching.fuzzy.predict import lookup_address_rapidfuzz
from glam.matching.vector.model import make_vectors
from glam.parsing.parsed_address import ParsedAddresses
from glam.types import NZSA


def nearest_neighbours(
    addresses: ParsedAddresses,
    trees: dict[int, cKDTree],
    batch_size: int = 1028,
    k: int = 100,
) -> pd.DataFrame:
    # vectorise address and create batches
    df = pd.DataFrame({"parsed_address": addresses})
    df["embedding"] = [
        make_vectors(addy, postcode=NZSA.postcodes) for addy in addresses
    ]
    df["embedding_dim"] = df["embedding"].apply(len)  # type: ignore
    df["batch"] = (
        df.groupby("embedding_dim")["parsed_address"].cumcount() // batch_size  # type: ignore
    )

    # group by batch
    grouped = df.groupby(["embedding_dim", "batch"])  # type: ignore
    n = len(grouped)  # type: ignore
    N = len(addresses)

    def hybrid_match(embeddings: pd.DataFrame, tree: cKDTree):
        parsed_addresses = embeddings["parsed_address"].to_list()

        # gets a shortlist of candidates
        dists, idxs = tree.query(embeddings["embedding"].to_list(), k)

        res = [
            lookup_address_rapidfuzz(parsed_addresses[i], NZSA.df.iloc[idxs[i]].copy())
            for i in range(len(embeddings))
        ]
        return res

    # match addresses
    matches = [
        zip(
            embeddings.index,
            hybrid_match(embeddings, trees[embedding_dim]),
        )
        for (embedding_dim, batch), embeddings in tqdm(
            grouped, unit_scale=N / n, desc="Matching addresses"
        )
    ]

    return (
        pd.DataFrame(
            [
                (id, match, confidence)
                for query in matches
                for id, (match, confidence) in query
            ],
            columns=["id", "address_id", "confidence"],
            dtype=float,
        )
        .sort_values("id")
        .set_index("id", drop=True)
    ).reset_index()
