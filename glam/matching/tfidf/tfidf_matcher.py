import pickle
import string
from itertools import product
from pathlib import Path
from typing import final, override

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from glam import logs
from glam.matching.base_matcher import BaseMatcher
from glam.parsing.parsed_address import ParsedAddresses
from glam.types import NZSA, Addresses, AddressIDs, Confidences

alphabet = string.ascii_lowercase + string.digits + " "
vocab = [x + y for x, y in product(alphabet, alphabet)]

logger = logs.get_logger()


@final
class TFIDFMatcher(BaseMatcher):
    requires_parser = False
    type = "TFIDF"
    vectoriser: TfidfVectorizer
    tfidf_matrix: sparse.csr_matrix
    idxmap: dict[int, int]

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir)

    @override
    def build_dependencies(
        self,
    ) -> None:
        super().build_dependencies()

        if NZSA.df is None:
            raise ValueError("NZSA data not loaded")

        if NZSA.postcodes:
            for_tfidf = (
                NZSA.df["full_address_ascii"] + " " + NZSA.df["postcode"].fillna("")
            )
        else:
            for_tfidf = NZSA.df["full_address_ascii"]

        # Create the TF-IDF vectorizer with character-level bigrams
        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 2), vocabulary=vocab
        )

        # add an idx map
        NZSA.df["address_id"].reset_index().to_csv(
            self.matcher_data / "idxmap.csv", index=False
        )

        tfidf_matrix = vectorizer.fit_transform(for_tfidf)
        sparse.save_npz(self.matcher_data / "tfidf_matrix.npz", tfidf_matrix)

        with open(self.matcher_data / "tfidf_vectoriser.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

    def load_dependencies(self) -> None:
        if not self.check_build():
            raise ValueError("TFIDF data not built")

        self.idxmap = (
            pd.read_csv(self.matcher_data / "idx_map.csv")  # type: ignore
            .set_index("index")["address_id"]
            .to_dict()
        )

        self.tfidf_matrix = sparse.load_npz(self.matcher_data / "tfidf_matrix.npz")

        with open(self.matcher_data / "tfidf_vectoriser.pkl", "rb") as f:
            self.vectoriser = pickle.load(f)

    def check_build(self) -> bool:
        matrix_check = self.matcher_data / "tfidf_matrix.npz"
        vectoriser_check = self.matcher_data / "tfidf_vectoriser.pkl"
        idxmap_check = self.matcher_data / "idxmap.csv"
        return matrix_check.exists() & vectoriser_check.exists() & idxmap_check.exists()

    @override
    def _match_unparsed_addresses(
        self, addresses: Addresses, batch_size: int = 128
    ) -> tuple[AddressIDs, Confidences]:
        address_ids = []
        confidences = []

        N = len(addresses)
        n = N // batch_size + (N % batch_size > 0)

        for start in tqdm(range(0, len(addresses), batch_size), unit_scale=N / n):
            end = start + batch_size
            batch = addresses[start:end]

            query = self.vectoriser.transform(batch)
            similarity_scores = cosine_similarity(query, self.tfidf_matrix)
            best_match_index = similarity_scores.argmax(axis=1)

            address_ids_confidences = [
                (self.idxmap[bm], similarity_scores[i][bm])
                for i, bm in enumerate(best_match_index)
            ]

            new_ids, new_confs = zip(*address_ids_confidences)

            address_ids.extend(new_ids)
            confidences.extend(new_confs)

        return address_ids, confidences

    @override
    def _match_parsed_addresses(
        self, addresses: ParsedAddresses
    ) -> tuple[AddressIDs, Confidences]:
        raise NotImplementedError("TFIDF matcher requires unparsed addresses")
