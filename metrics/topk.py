from typing import Any, List, Tuple
import torch

from metrics.metric import Metric


class TopKMetric(Metric):
    """
    Metric to compute the top-k accuracy based on embeddings.
    """

    def __init__(self, k=5):
        self.k = k

    def compute(
        self, embeddings: List[Tuple[List[torch.Tensor], torch.Tensor]]
    ) -> Tuple[float, Any]:
        """Compute the top-k accuracy.

        Args:
            embeddings (List[Tuple[List[torch.Tensor], torch.Tensor]]): list of (text_embedding[], image_embedding) pairs.

        Returns:
            Tuple[float, Any]: Top-k accuracy score and additional info.
        """

        return top_k_score(embeddings, self.k)


def top_k_similarities(embeddings, query_embedding, k=5):
    """
    Compute the top-k most similar embeddings to the query_embedding.

    Args:
        embeddings (torch.Tensor): Tensor of shape (N, D) where N is the number of embeddings and D is the embedding dimension.
        query_embedding (torch.Tensor): Tensor of shape (D,) representing the query embedding.
        k (int): Number of top similar embeddings to return.

    Returns:
        List[Tuple[int, float]]: List of tuples containing the index and similarity score of the top-k most similar embeddings.
    """
    # Compute cosine similarities
    similarities = torch.nn.functional.cosine_similarity(
        embeddings, query_embedding.unsqueeze(0), dim=1
    )

    # Get top-k indices
    top_k_indices = similarities.topk(k).indices

    # Return list of (index, similarity) tuples
    return [(idx.item(), similarities[idx].item()) for idx in top_k_indices]


def top_k_score(embedding_pairs, k=5):
    """
    Given a list of (text_embedding[], image_embedding) pairs, return the percentage of texts that are in the top-k most similar to their corresponding image embeddings.
    """
    correct_count = 0
    total_count = len(embedding_pairs)

    all_text_embeddings = [
        text_emb
        for text_embeddings, _ in embedding_pairs
        for text_emb in text_embeddings
    ]
    text_emb_tensor = torch.stack(all_text_embeddings)  # Shape: (num_texts, D)

    sofar = 0
    for correct_texts, image_embedding in embedding_pairs:
        # Get top-k similar text embeddings to the image embedding
        top_k = top_k_similarities(text_emb_tensor, image_embedding, k)

        # Check if the correct text embedding is in the top-k
        correct_indices = set(range(sofar, sofar + len(correct_texts)))
        top_k_indices = set(idx for idx, _ in top_k)

        if correct_indices.intersection(top_k_indices):
            correct_count += 1

        sofar += len(correct_texts)
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return accuracy, {"correct": correct_count, "total": total_count}


if __name__ == "__main__":
    # Simple test
    emb1 = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    query = torch.tensor([1.0, 0.0])
    top_k = top_k_similarities(emb1, query, k=2)
    print(
        "Top-k similarities:", top_k
    )  # Expected: indices 0 and 1 with highest similarities

    embeddings = [
        (
            [
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.9, 0.1]),
                torch.tensor([0.0, 1.0]),
            ],
            torch.tensor([1.0, 0.0]),
        )
    ]

    score = top_k_score(embeddings, k=1)
    print("Top-k score:", score)  # Expected: 1.0
