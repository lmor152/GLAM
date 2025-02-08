from typing import Type

from glam.matching.base_matcher import BaseMatcher
import importlib

matchers: dict[str, str] = {
    "fuzzy": "glam.matching.fuzzy.fuzzy_matcher.FuzzyMatcher",
    "vector": "glam.matching.vector.vector_matcher.VectorMatcher",
    "hybrid-fuzzy": "glam.matching.hybrid_fuzzy.hybrid_fuzzy_matcher.HybridFuzzyMatcher",
    "tfidf": "glam.matching.tfidf.tfidf_matcher.TFIDFMatcher",
    "hybrid-tfidf": "glam.matching.hybrid_tfidf.hybrid_tfidf_matcher.HybridTFIDFMatcher",
    "embedding": "glam.matching.embedding.embedding_matcher.EmbeddingMatcher",
}

def get_matcher(matcher_name: str) -> Type[BaseMatcher]:
    matcher_path = matchers.get(matcher_name)
    if matcher_path is None:
        raise ValueError(f"Unknown matcher: {matcher_name}")

    # Dynamically import the matcher module
    module_name, class_name = matcher_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


