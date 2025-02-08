import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def nearest_neighbours(
    embeddings: np.array, tree: cKDTree, batch_size=256
) -> tuple[np.array, np.array]:
    num_sections = max(1, len(embeddings) // batch_size)
    batches = np.array_split(embeddings, num_sections)

    all_matches = []
    all_distances = []
    for batch in tqdm(batches, unit_scale=len(embeddings) / len(batches)):
        distances, indices = tree.query(batch, k=1)
        all_matches.extend(indices)
        all_distances.extend(distances.flatten())

    return np.array(all_matches), np.array(all_distances)
