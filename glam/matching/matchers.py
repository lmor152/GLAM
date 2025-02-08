from typing import Type

from glam.matching.base_matcher import BaseMatcher
from glam.matching.embedding.embedding_matcher import EmbeddingMatcher
from glam.matching.fuzzy.fuzzy_matcher import FuzzyMatcher
from glam.matching.hybrid_fuzzy.hybrid_fuzzy_matcher import HybridFuzzyMatcher
from glam.matching.hybrid_tfidf.hybrid_tfidf_matcher import HybridTFIDFMatcher
from glam.matching.tfidf.tfidf_matcher import TFIDFMatcher
from glam.matching.vector.vector_matcher import VectorMatcher

matchers: dict[str, Type[BaseMatcher]] = {
    "fuzzy": FuzzyMatcher,
    "vector": VectorMatcher,
    "hybrid-fuzzy": HybridFuzzyMatcher,
    "tfidf": TFIDFMatcher,
    "hybrid-tfidf": HybridTFIDFMatcher,
    "embedding": EmbeddingMatcher,
}
