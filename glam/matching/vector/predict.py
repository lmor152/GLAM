import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from glam.matching.vector.model import make_vectors
from glam.parsing.parsed_address import ParsedAddresses
from glam.types import NZSA


def nearest_neighbours(
    addresses: ParsedAddresses, trees: dict[int, cKDTree], batch_size: int = 1028
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

    # match addresses
    matches = [
        zip(
            embeddings.index,
            query_tree(embeddings["embedding"].to_list(), trees[embedding_dim]),
        )
        for (embedding_dim, batch), embeddings in tqdm(
            grouped, unit_scale=N / n, desc="Matching addresses"
        )
    ]

    return (
        pd.DataFrame(
            [(id, match, dist) for query in matches for id, (dist, match) in query],
            columns=["id", "match", "embedding_distance"],
        )
        .sort_values("id")
        .set_index("id", drop=True)
    ).reset_index()


def query_tree(
    embeddings: list[float], tree: cKDTree | None
) -> list[tuple[float, int]]:
    if tree is None:
        return list(zip([-1] * len(embeddings), [-1] * len(embeddings)))

    dist, idx = tree.query(embeddings)

    return list(zip(dist / tree.m, idx))  # type: ignore
