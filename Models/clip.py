# This defines a CLIP-like interface for models to ascribe to, but does not actually implement CLIP itself.
# This is used for standardizing the interface for clip variants in testing.

from typing import List, List, Union
import torch
import torch.nn as nn


class ClipModel(nn.Module):
    def encode_text(
        self,
        texts: Union[str, List[str]],
        requires_grad: bool = True,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Encode text strings to embeddings.

        Args:
            texts: Single text string or list of text strings
            requires_grad: Whether to compute gradients (default: True)
            normalize: Whether to normalize embeddings to unit sphere (default: False)

        Returns:
            Text embeddings
            Shape: [batch_size, 512]
        """
        ...

    def encode_images(
        self,
        image_paths: Union[str, List[str]],
        requires_grad: bool = True,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Encode images from file paths to embeddings.

        Args:
            image_paths: Single image path or list of image paths
            requires_grad: Whether to compute gradients (default: True)
            normalize: Whether to normalize embeddings to unit sphere (default: False)

        Returns:
            Image embeddings
            Shape: [batch_size, num_features]
        """
        ...

    ...
