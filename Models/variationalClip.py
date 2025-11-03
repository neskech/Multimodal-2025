"""
CLIP embedding extraction functionality for von Mises-Fisher mixture modeling.
"""

import torch
import clip
from torch import nn
from PIL import Image
from typing import List, Literal, TypeAlias, Union

from Models.clipInterface import ClipInterface


# Global constant for model configuration
MODEL_NAME = "ViT-B/32"
CLIP_EMBEDDING_DIM = 512
ModelType: TypeAlias = Literal["Spherical", "Gaussian"]


class VariationalCLIPModel(ClipInterface):
    """
    Variational CLIP model that outputs von Mises-Fisher distribution parameters.
    Hard-coded to use ViT-B/32 architecture.
    Modified to output mean direction (512D) and concentration parameter (1D).
    """

    def __init__(self, model_type: ModelType, device: str | None = None):
        """
        Initialize CLIP model.

        Args:
            model_type: Type of model ('Spherical' or 'Gaussian')
                If spherical, outputs a scalar concentration parameter.
                If gaussian, outputs a log-variance parameter.
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = MODEL_NAME

        # Load CLIP model
        self.model, self.preprocess = clip.load(MODEL_NAME, device=self.device)

        # HACK: Clip initializes their projection as nn.Parameter(scale * torch.randn(width, output_dim))
        # In order to do the same, we need to extract that scale parameter. The scale is derived from
        # the 'visual.width' parameter, which is used inside of this convolutional layer.
        vision_width = self.model.visual.conv1.out_channels
        scale = vision_width**-0.5
        self.mean_image_projection = nn.Parameter(
            scale * torch.randn(vision_width, CLIP_EMBEDDING_DIM)
        )
        if model_type == "Spherical":
            self.var_image_projection = nn.Parameter(
                scale * torch.randn(vision_width, 1)
            )
        else:
            self.var_image_projection = nn.Parameter(
                scale * torch.randn(vision_width, CLIP_EMBEDDING_DIM)
            )

        # HACK: Clip uses the transformer.width parameter to initialize the projection layer
        # As such, we need to extract that width parameter.
        transformer_width = self.model.transformer.width
        self.mean_text_projection = nn.Parameter(
            scale * torch.randn(transformer_width, CLIP_EMBEDDING_DIM)
        )
        if model_type == "Spherical":
            self.var_text_projection = nn.Parameter(
                scale * torch.randn(transformer_width, 1)
            )
        else:
            self.var_text_projection = nn.Parameter(
                scale * torch.randn(transformer_width, CLIP_EMBEDDING_DIM)
            )

        # CLIP only seems to do this initialization for the text projection
        nn.init.normal_(self.mean_text_projection, std=transformer_width**-0.5)
        nn.init.normal_(self.var_text_projection, std=vision_width**-0.5)

        # Add an additional class token for the concetration parameter. Will
        # be appened to the end of the image patch sequence that feeds into the VIT
        self.image_concentration_embedding = nn.Parameter(
            scale * torch.randn(vision_width)
        )

    def encode_image_internal(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: Highly copy and pasted from the VIT forward function in CLIP
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        mean_embedding: torch.Tensor = self.model.visual.class_embedding.to(x.dtype)  # type: ignore
        zeros = torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        mean_embedding_broadcasted = mean_embedding + zeros

        concentration_embedding = self.image_concentration_embedding.to(x.dtype)
        concentration_embedding_broadcasted = concentration_embedding + zeros

        # shape = [*, grid ** 2 + 2, width]
        x = torch.cat(
            [mean_embedding_broadcasted, x, concentration_embedding_broadcasted], dim=1
        )

        x = x + self.model.visual.positional_embedding.to(x.dtype)  # type: ignore
        x = self.model.visual.ln_pre(x)  # type: ignore

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)  # type: ignore
        x = x.permute(1, 0, 2)  # LND -> NLD

        mean_embedding = self.model.visual.ln_post(x[:, 0, :])  # type: ignore
        concentration_embedding = self.model.visual.ln_post(x[:, -1, :])  # type: ignore

        mean_embedding = mean_embedding @ self.mean_image_projection
        concentration_embedding = concentration_embedding @ self.var_image_projection
        return mean_embedding, concentration_embedding

    def encode_text_internal(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement
        pass

    def get_logits_scale(self) -> torch.Tensor:
        """Get the logits scale parameter."""
        return self.model.logit_scale

    def encode_image_tensors(
        self, image_tensors: torch.Tensor, requires_grad: bool = True
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
            mean, var = self.encode_image_internal(image_tensors)
        else:
            with torch.no_grad():
                mean, var = self.encode_image_internal(image_tensors)

        return mean, var

    def encode_text_tokens(
        self, text_tokens: torch.Tensor, requires_grad: bool = True
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

        return mean_direction, concentration

    def encode_text(
        self, texts: Union[str, List[str]], requires_grad: bool = True
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

        return self.encode_text_tokens(text_tokens, requires_grad=requires_grad)

    def encode_images(
        self, image_paths: Union[str, List[str]], requires_grad: bool = True
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

        return self.encode_image_tensors(image_tensors, requires_grad=requires_grad)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of CLIP embeddings."""
        # CLIP ViT-B/32 has 512-dimensional embeddings
        return CLIP_EMBEDDING_DIM
