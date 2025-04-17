import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_distance(embedding1, embedding2):
    """Calculates the cosine distance between two embeddings."""
    # Ensure inputs are numpy arrays
    embedding1 = np.asarray(embedding1)
    embedding2 = np.asarray(embedding2)

    # Check for zero vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        if norm1 == 0 and norm2 == 0:
            return 0.0 # Distance between two zero vectors is 0
        else:
            return 1.0 # Distance between zero vector and non-zero vector is 1 (max cosine distance)

    # Reshape for cosine_similarity function (expects 2D arrays)
    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)

    # Calculate cosine similarity and convert to distance
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    # Clip similarity to [-1, 1] range due to potential floating point inaccuracies
    similarity = np.clip(similarity, -1.0, 1.0)
    return 1 - similarity