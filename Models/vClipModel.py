"""
CLIP embedding extraction functionality for von Mises-Fisher mixture modeling.
"""

import torch
import clip
from torch import nn
from PIL import Image
from typing import List, Union

from Models.clipModel import ClipInterface


# Global constant for model configuration
MODEL_NAME = "ViT-B/32"
CLIP_EMBEDDING_DIM = 512


class VariationalCLIPModel(ClipInterface):
    """
    Variational CLIP model that outputs von Mises-Fisher distribution parameters.
    Hard-coded to use ViT-B/32 architecture.
    Modified to output mean direction (512D) and concentration parameter (1D).
    """

    def __init__(self, device: str | None = None):
        """
        Initialize CLIP model.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = MODEL_NAME

        # Load CLIP model
        self.model, self.preprocess = clip.load(MODEL_NAME, device=self.device)

        # Modify vision encoder to output distribution parameters
        # HACK: Clip initializes their projection as nn.Parameter(scale * torch.randn(width, output_dim))
        # In order to do the same, we need to extract that scale parameter. The scale is derived from
        # the 'visual.width' parameter, which is used inside of this convolutional layer.
        vision_width = self.model.visual.conv1.out_channels
        scale = vision_width**-0.5

        output_dim = CLIP_EMBEDDING_DIM + 1
        self.model.visual.proj = nn.Parameter(
            scale * torch.randn(vision_width, output_dim)
        )

        # Modify text encoder to output distribution parameters
        # HACK: Clip uses the transformer.width parameter to initialize the projection layer
        # As such, we need to extract that width parameter.
        transformer_width = self.model.transformer.width
        output_dim = CLIP_EMBEDDING_DIM + 1
        self.model.text_encoder.proj = nn.Parameter(
            torch.empty(transformer_width, output_dim)
        )

    def encode_image_tensors(
        self,
        image_tensors: torch.Tensor,
        requires_grad: bool = True,
        normalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image tensors to von Mises-Fisher distribution parameters.

        Args:
            image_tensors: Batch of image tensors
                          Shape: [batch_size, 3, 224, 224]
            requires_grad: Whether to compute gradients (default: True)
            normalize: Whether to normalize mean direction to unit sphere (default: False)

        Returns:
            Tuple of (mean_direction, concentration)
            mean_direction: Shape [batch_size, 512]
            concentration: Shape [batch_size]
        """
        if requires_grad:
            # Get full output from modified model.encode_image (512 + 1 dimensions)
            full_output = self.model.encode_image(image_tensors)
        else:
            with torch.no_grad():
                full_output = self.model.encode_image(image_tensors)

        # Extract mean direction (first 512 elements) and concentration (last element)
        mean_direction = full_output[:, :-1]  # Shape: [batch_size, 512]
        concentration = full_output[:, -1]  # Shape: [batch_size]

        if normalize:
            mean_direction = mean_direction / mean_direction.norm(dim=1, keepdim=True)

        return mean_direction, concentration

    def encode_text_tokens(
        self,
        text_tokens: torch.Tensor,
        requires_grad: bool = True,
        normalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode tokenized text to von Mises-Fisher distribution parameters.

        Args:
            text_tokens: Batch of tokenized text tensors
                        Shape: [batch_size, 77] (context_length)
            requires_grad: Whether to compute gradients (default: True)
            normalize: Whether to normalize mean direction to unit sphere (default: False)

        Returns:
            Tuple of (mean_direction, concentration)
            mean_direction: Shape [batch_size, 512]
            concentration: Shape [batch_size]
        """
        if requires_grad:
            # Get full output from modified model.encode_text (512 + 1 dimensions)
            full_output = self.model.encode_text(text_tokens)
        else:
            with torch.no_grad():
                full_output = self.model.encode_text(text_tokens)

        # Extract mean direction (first 512 elements) and concentration (last element)
        mean_direction = full_output[:, :-1]  # Shape: [batch_size, 512]
        concentration = full_output[:, -1]  # Shape: [batch_size]

        if normalize:
            mean_direction = mean_direction / mean_direction.norm(dim=1, keepdim=True)

        return mean_direction, concentration

    def encode_text(
        self,
        texts: Union[str, List[str]],
        requires_grad: bool = True,
        normalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text strings to von Mises-Fisher distribution parameters.

        Args:
            texts: Single text string or list of text strings
            requires_grad: Whether to compute gradients (default: True)
            normalize: Whether to normalize mean direction to unit sphere (default: False)

        Returns:
            Tuple of (mean_direction, concentration)
            mean_direction: Shape [batch_size, 512]
            concentration: Shape [batch_size]
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize text
        text_tokens = clip.tokenize(texts, truncate=True)
        text_tokens = text_tokens.to(self.device)

        return self.encode_text_tokens(
            text_tokens, requires_grad=requires_grad, normalize=normalize
        )

    def encode_images(
        self,
        image_paths: Union[str, List[str]],
        requires_grad: bool = True,
        normalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images from file paths to von Mises-Fisher distribution parameters.

        Args:
            image_paths: Single image path or list of image paths
            requires_grad: Whether to compute gradients (default: True)
            normalize: Whether to normalize mean direction to unit sphere (default: False)

        Returns:
            Tuple of (mean_direction, concentration)
            mean_direction: Shape [batch_size, 512]
            concentration: Shape [batch_size]
        """
        # Handle single path input
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Load and preprocess images
        image_tensors = []
        for image_path in image_paths:
            try:
                # Load image
                image = Image.open(image_path).convert("RGB")
                # Use CLIP's preprocessing
                image_tensor = self.preprocess(image).unsqueeze(0)
                image_tensors.append(image_tensor)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Return a black image as fallback
                image_tensors.append(torch.zeros(1, 3, 224, 224))

        # Concatenate all images
        image_tensors = torch.cat(image_tensors, dim=0).to(self.device)

        return self.encode_image_tensors(
            image_tensors, requires_grad=requires_grad, normalize=normalize
        )

    def get_embedding_dimension(self) -> int:
        """Get the dimension of CLIP embeddings."""
        # CLIP ViT-B/32 has 512-dimensional embeddings
        return CLIP_EMBEDDING_DIM
