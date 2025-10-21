"""
CLIP embedding extraction functionality for von Mises-Fisher mixture modeling.
"""

import torch
import clip
from torch import nn
from typing import Tuple
from clip.model import VisionTransformer
import copy


# Global constant for model configuration
MODEL_NAME = "ViT-B/32"
NUM_SHARED_LAYERS = 4


class AlignCLIPModel(nn.Module):
    """
    Align CLIP embedding extractor as a PyTorch module.
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

        # Create shared transformer blocks
        self._create_and_assign_shared_layers()

    def _create_and_assign_shared_layers(self):
        """
        Modifies the loaded CLIP model in-place to use shared transformer layers.
        """
        if not isinstance(self.model.visual, VisionTransformer):
            raise TypeError("AlignCLIPModel only supports VisionTransformer-based CLIP models.")

        visual_transformer_blocks = list(self.model.visual.transformer.resblocks)
        text_transformer_blocks = list(self.model.transformer.resblocks)

        assert NUM_SHARED_LAYERS <= len(
            text_transformer_blocks
        ) and NUM_SHARED_LAYERS <= len(
            visual_transformer_blocks
        ), "Cannot share more layers than exist."

        # Create shared layers, initialized from the text encoder's weights
        shared_blocks = nn.ModuleList(
            [copy.deepcopy(layer) for layer in text_transformer_blocks[-NUM_SHARED_LAYERS:]]
        )

        # Replace the final layers in both encoders with the same shared blocks
        self.model.visual.transformer.resblocks = nn.ModuleList( # type: ignore
            list(visual_transformer_blocks[:-NUM_SHARED_LAYERS]) + list(shared_blocks)
        )
        self.model.transformer.resblocks = nn.ModuleList( # type: ignore
            list(text_transformer_blocks[:-NUM_SHARED_LAYERS]) + list(shared_blocks)
        )

    def encode_image_features(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        Manually replicates the vision encoder forward pass to get pre-projection features.
        """
        if not isinstance(self.model.visual, VisionTransformer):
            raise TypeError("encode_image_features only supports VisionTransformer-based CLIP models.")

        x = image_tensors.type(self.model.dtype)

        x = self.model.visual.conv1(x)  # [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # [*, grid ** 2, width]

        # Prepend class embedding
        class_embedding= self.model.visual.class_embedding.to(x.dtype)
        zeros = torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([class_embedding + zeros, x], dim=1)

        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        image_features = self.model.visual.ln_post(x[:, 0, :])

        return image_features

    def encode_text_features(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Manually replicates the text encoder forward pass to get pre-projection features.
        """
        x = self.model.token_embedding(text_tokens).type(self.model.dtype)

        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        # Get the features corresponding to the EOT token
        text_features = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]

        return text_features

    def forward(
        self,
        image_tensors: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(self.model.visual, VisionTransformer):
            raise TypeError("AlignCLIPModel only supports VisionTransformer-based CLIP models.")

        image_features = self.encode_image_features(image_tensors)
        text_features = self.encode_text_features(text_tokens)

        # Apply the final projections to get the embeddings
        image_embeds = image_features @ self.model.visual.proj
        text_embeds = text_features @ self.model.text_projection

        return image_embeds, text_embeds, image_features, text_features


