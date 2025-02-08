import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from glam.matching.vector.model import make_vectors
from glam.parsing.parsed_address import ParsedAddress, ParsedAddresses
from glam.types import NZSA


def nearest_neighbours(
    addresses: ParsedAddresses,
    trees: dict[int, cKDTree],
    vectoriser: TfidfVectorizer,
    tfidf_matrix: csr_matrix,
    idxmap: dict[int, int],
    batch_size: int = 1028,
    k: int = 100,
) -> pd.DataFrame:
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

    def get_address_tfidf(query: ParsedAddress, idxs: list[int]):
        similarities = cosine_similarity(query, tfidf_matrix[idxs])
        argmax = similarities.argmax()
        return idxmap[idxs[argmax]], similarities[0][argmax]

    def hybrid_match(embeddings: pd.DataFrame, tree: cKDTree):
        parsed_addresses = embeddings["parsed_address"].to_list()

        # gets a shortlist of candidates
        _, idxs = tree.query(embeddings["embedding"].to_list(), k)

        queries = vectoriser.transform(
            [pa.format_address(human=True) for pa in parsed_addresses]
        )

        res = [get_address_tfidf(queries[i], idxs[i]) for i in range(len(embeddings))]

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
