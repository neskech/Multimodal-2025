from typing import List, Tuple

from torch import Tensor
import torch

from metrics.metric import Metric


class EntropyMetric(Metric):
    """
    Metric to compute the entropy of embeddings.

    Since entropy is infeasible in high dimension unit sphere, we
    use a proxy -- the average angle between each embedding and its k'th nearest neighbor.
    """

    def __init__(self, k=5):
        self.k = k

    def compute(self, embeddings: List[Tuple[List[Tensor], Tensor]]) -> float:
        """Compute the entropy metric.

        Args:
            embeddings (List[Tuple[List[torch.Tensor], torch.Tensor]]): list of (text_embedding[], image_embedding) pairs.

        Returns:
            float: Entropy score.
        """
        all_embeddings = torch.cat(
            [torch.stack(text_embs + [img_emb]) for text_embs, img_emb in embeddings],
            dim=0,
        )
        entropy_score = average_kth_neighbor_angle(all_embeddings, self.k)
        return entropy_score


def average_kth_neighbor_angle(embeddings: Tensor, k=5, batch_size=1000) -> float:
    """
    Compute the average angle to the k'th nearest neighbor for each embedding.
    Uses batching to avoid creating a full N×N similarity matrix in memory.

    Args:
        embeddings (torch.Tensor): Tensor of shape (N, D) where N is the number of embeddings and D is the embedding dimension.
        k (int): The k'th nearest neighbor to consider.
        batch_size (int): Number of embeddings to process at once to control memory usage.

    Returns:
        float: Average angle to the k'th nearest neighbor in radians.
    """
    n = embeddings.size(0)
    device = embeddings.device

    # Normalize embeddings for efficient cosine similarity computation
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    all_kth_angles = []

    # Process in batches to avoid memory issues
    for i in range(0, n, batch_size):
        batch_end = min(i + batch_size, n)
        batch_embeddings = embeddings_norm[i:batch_end]

        # Compute similarities for this batch against all embeddings
        # Shape: (batch_size, N)
        similarities = torch.mm(batch_embeddings, embeddings_norm.t())

        # Set self-similarities to -inf to ignore them
        for j in range(batch_embeddings.size(0)):
            similarities[j, i + j] = -float("inf")

        # Get the k'th nearest neighbor similarities
        kth_similarities, _ = similarities.topk(k, dim=1)

        # Convert cosine similarities to angles
        kth_angles = torch.acos(torch.clamp(kth_similarities[:, -1], -1.0, 1.0))
        all_kth_angles.append(kth_angles)

    # Concatenate all angles and compute mean
    all_kth_angles = torch.cat(all_kth_angles)
    return all_kth_angles.mean().item()
