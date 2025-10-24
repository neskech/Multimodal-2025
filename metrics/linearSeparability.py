from typing import List, Tuple
import torch.nn as nn
import torch
from tqdm import tqdm

from metrics.metric import Metric


class SeperabilityMetric(Metric):
    def __init__(self, n_epochs=100, lr=1e-3):
        """Initialize the SeparabilityMetric.

        Args:
            n_epochs (int, optional): Number of training epochs. Defaults to 100.
            lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        """
        self.n_epochs = n_epochs
        self.lr = lr

    def compute(self, embeddings: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Compute the linear separability metric.

        Args:
            embeddings (List[Tuple[torch.Tensor, torch.Tensor]]): list of (image_embedding, text_embedding) pairs.

        Returns:
            float: Linear separability accuracy.
        """
        image_embeddings = torch.cat([img_emb for img_emb, _ in embeddings], dim=0)
        text_embeddings = torch.cat([txt_emb for _, txt_emb in embeddings], dim=0)
        accuracy = linear_separability(
            image_embeddings,
            text_embeddings,
            num_epochs=self.n_epochs,
            learning_rate=self.lr,
        )
        return accuracy


def linear_separability(
    image_embeddings, text_embeddings, num_epochs=100, learning_rate=1e-3
):
    """
    Train a linear classifier to distinguish between image and text embeddings, and report the accuracy.

    Args:
        image_embeddings (torch.Tensor): Tensor of shape (N, D) for image embeddings.
        text_embeddings (torch.Tensor): Tensor of shape (N, D) for text embeddings.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        float: Accuracy of the classifier on the given set.
    """
    # Combine image and text embeddings and detach to avoid backprop into upstream graphs
    embeddings = torch.cat([image_embeddings, text_embeddings], dim=0).detach()
    device = image_embeddings.device
    embeddings = embeddings.to(device)
    labels = torch.cat(
        [
            torch.zeros(image_embeddings.size(0), dtype=torch.long, device=device),
            torch.ones(text_embeddings.size(0), dtype=torch.long, device=device),
        ],
        dim=0,
    )

    # Train a linear classifier (on the same device as the embeddings)
    classifier = nn.Linear(embeddings.size(1), 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        classifier.train()
        outputs = classifier(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))
        optimizer.step()
        optimizer.zero_grad()

    # Evaluate the classifier
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(embeddings)
        preds = outputs.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()

    return accuracy
