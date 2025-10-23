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


def average_kth_neighbor_angle(embeddings: Tensor, k=5) -> float:
    """
    Compute the average angle to the k'th nearest neighbor for each embedding.

    Args:
        embeddings (torch.Tensor): Tensor of shape (N, D) where N is the number of embeddings and D is the embedding dimension.
        k (int): The k'th nearest neighbor to consider.

    Returns:
        float: Average angle to the k'th nearest neighbor in radians.
    """
    # Compute cosine similarities
    similarities = torch.nn.functional.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
    )  # Shape: (N, N)

    # Set self-similarities to -inf to ignore them
    similarities.fill_diagonal_(-float("inf"))

    # Get the k'th nearest neighbor similarities
    kth_similarities, _ = similarities.topk(k, dim=1)

    # Convert cosine similarities to angles
    kth_angles = torch.acos(torch.clamp(kth_similarities[:, -1], -1.0, 1.0))

    # Return average angle
    return kth_angles.mean().item()
