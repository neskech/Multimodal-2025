"""
CLIP embedding extraction functionality for von Mises-Fisher mixture modeling.
"""

import torch
import clip
import numpy as np
from torch import nn
from PIL import Image
from typing import List, Literal, TypeAlias, Union

from Models.clipInterface import ClipInterface


# Global constant for model configuration
MODEL_NAME = "ViT-B/32"
CLIP_EMBEDDING_DIM = 512
ModelType: TypeAlias = Literal["Spherical", "Gaussian"]


def _build_attention_mask(seq_length: int):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(seq_length, seq_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask

class VariationalCLIPModel(ClipInterface):
    """
    Variational CLIP model that outputs von Mises-Fisher distribution parameters.
    Hard-coded to use ViT-B/32 architecture.
    Modified to output mean direction (512D) and concentration parameter (1D).
    """

    def __init__(self, model_type: ModelType, device: str | None = None, use_pretrained: bool = True,
                 min_concentration: float = 10.0, initial_concentration: float = 200.0):
        """
        Initialize CLIP model.

        Args: 
            model_type: Type of model ('Spherical' or 'Gaussian')
                If spherical, outputs a scalar concentration parameter.
                If gaussian, outputs a log-variance parameter.
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            use_pretrained: If True, initialize projections and positional embeddings 
                from CLIP's pretrained weights. If False, randomly initialize everything.
            min_concentration: Minimum concentration value (default: 10.0)
                Ensures distributions don't collapse to uniform.
            initial_concentration: Target initial concentration value (default: 200.0)
                Used to initialize the learnable scale parameters.
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = MODEL_NAME
        self.model_type = model_type
        self.min_concentration = min_concentration
        self.initial_concentration = initial_concentration

        # Load CLIP model (always load to get architecture, but may reinitialize weights)
        self.model, self.preprocess = clip.load(MODEL_NAME, device=self.device)

        # Extract width parameters for initialization
        vision_width = self.model.visual.conv1.out_channels
        transformer_width = self.model.transformer.width
        scale = vision_width**-0.5
        text_scale = transformer_width**-0.5

        # Get sequence lengths before modifying
        visual_seq_length = self.model.visual.positional_embedding.shape[0]
        text_seq_length = self.model.positional_embedding.shape[0]

        if use_pretrained:
            # ===== PRETRAINED INITIALIZATION =====
            # Copy pretrained weights and extend positional embeddings
            
            # IMAGE ENCODER
            self.mean_image_projection = nn.Parameter(
                self.model.visual.proj.data.clone()
            )
            # Extend visual positional embeddings: copy pretrained + add 1 new random position
            old_visual_pos = self.model.visual.positional_embedding.data.clone()
            new_visual_pos = torch.zeros(visual_seq_length + 1, vision_width, device=self.device)
            new_visual_pos[:visual_seq_length, :] = old_visual_pos
            new_visual_pos[visual_seq_length, :] = scale * torch.randn(vision_width)
            self.model.visual.positional_embedding = nn.Parameter(new_visual_pos)

            # TEXT ENCODER
            self.mean_text_projection = nn.Parameter(
                self.model.text_projection.data.clone()
            )
            # Extend text positional embeddings: copy pretrained + add 1 new random position
            old_text_pos = self.model.positional_embedding.data.clone()
            new_text_pos = torch.zeros(text_seq_length + 1, transformer_width, device=self.device)
            new_text_pos[:text_seq_length, :] = old_text_pos
            new_text_pos[text_seq_length, :] = text_scale * torch.randn(transformer_width)
            self.model.positional_embedding = nn.Parameter(new_text_pos)
        else:
            # ===== RANDOM INITIALIZATION =====
            # Randomly initialize everything from scratch
            
            # IMAGE ENCODER - random projection
            self.mean_image_projection = nn.Parameter(
                scale * torch.randn(vision_width, CLIP_EMBEDDING_DIM)
            )
            # Random visual positional embeddings (extended length)
            self.model.visual.positional_embedding = nn.Parameter(
                scale * torch.randn(visual_seq_length + 1, vision_width)
            )

            # TEXT ENCODER - random projection
            self.mean_text_projection = nn.Parameter(
                text_scale * torch.randn(transformer_width, CLIP_EMBEDDING_DIM)
            )
            # Random text positional embeddings (extended length)
            self.model.positional_embedding = nn.Parameter(
                text_scale * torch.randn(text_seq_length + 1, transformer_width)
            )

            # Reinitialize all CLIP backbone weights randomly
            self._reinitialize_clip_weights()

        # ===== VARIANCE PROJECTIONS (always random, no pretrained equivalent) =====
        if model_type == "Spherical":
            # Use log space + learnable scale parameter approach
            # This works better with layer-normalized features:
            # - Projection learns relative differences in log space (small values, centered around 0)
            # - Learnable scale parameter controls global magnitude
            # - Final: concentration = exp(log_scale + log_concentration_raw) + min_concentration
            
            target_concentration_net = self.initial_concentration - self.min_concentration
            
            # Learnable scale parameters in log space (initialized to log(target))
            # These control the global magnitude of concentration
            self.log_concentration_scale_image = nn.Parameter(
                torch.tensor(np.log(target_concentration_net))  # log(190) ≈ 5.25
            )
            self.log_concentration_scale_text = nn.Parameter(
                torch.tensor(np.log(target_concentration_net))  # log(190) ≈ 5.25
            )
            
            # Projection learns relative differences (initialize small, centered around 0)
            # Since we're in log space, these can be small and still work with normalized inputs
            self.var_image_projection = nn.Parameter(
                scale * torch.randn(vision_width, 1)  # Small values around 0
            )
            self.var_text_projection = nn.Parameter(
                text_scale * torch.randn(transformer_width, 1)  # Small values around 0
            )
        else:
            self.var_image_projection = nn.Parameter(
                scale * torch.randn(vision_width, CLIP_EMBEDDING_DIM)
            )
            self.var_text_projection = nn.Parameter(
                text_scale * torch.randn(transformer_width, CLIP_EMBEDDING_DIM)
            )

        # Replace the attention masks with (seq length + 1) to account for the concentration token
        mask = _build_attention_mask(text_seq_length + 1)
        for block in self.model.transformer.resblocks:
            block.attn_mask = mask

        # ===== CONCENTRATION EMBEDDINGS (always random) =====
        # Initialize with smaller variance to encourage more stable learning
        self.image_concentration_embedding = nn.Parameter(
            scale * torch.randn(vision_width) * 0.1
        )
        self.text_concentration_embedding = nn.Parameter(
            text_scale * torch.randn(transformer_width) * 0.1
        )

    def _reinitialize_clip_weights(self):
        """Reinitialize all CLIP backbone weights randomly."""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze or unfreeze the CLIP backbone weights.
        
        When frozen, only the new variational layers are trained:
        - mean_image_projection, mean_text_projection
        - var_image_projection, var_text_projection
        - image_concentration_embedding, text_concentration_embedding
        - Extended positional embedding positions (last position only)
        
        Args:
            freeze: If True, freeze backbone. If False, unfreeze (allow training).
        """
        # Freeze/unfreeze all parameters in the CLIP backbone
        for param in self.model.parameters():
            param.requires_grad = not freeze
        
        # Always keep our new layers trainable (they're direct attributes, not in self.model)
        # These are already separate nn.Parameters so they won't be affected by the above loop
        
        # Log the status
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = total_params - trainable_params
        
        status = "frozen" if freeze else "unfrozen"
        print(f"Backbone {status}. Trainable: {trainable_params:,} / {total_params:,} params ({frozen_params:,} frozen)")
        
        return self

    def unfreeze_backbone(self):
        """Convenience method to unfreeze the backbone."""
        return self.freeze_backbone(freeze=False)

    def get_trainable_params(self):
        """Return only trainable parameters (useful for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

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

        if self.model_type == "Spherical":
            concentration_embedding = concentration_embedding.squeeze(-1)
            # Use log space + learnable scale: log_concentration = log_scale + projection_output
            # This works well with normalized features (projection learns relative differences)
            log_concentration_raw = concentration_embedding  # Already in log space from projection
            log_concentration = self.log_concentration_scale_image + log_concentration_raw

            # Clamp log_concentration to avoid overflow / NaNs in exp and downstream KL
            LOG_K_MIN = -5.0   # exp(-5)  ~ 0.0067
            LOG_K_MAX = 10.0   # exp(10)  ~ 22k
            log_concentration = torch.clamp(log_concentration, LOG_K_MIN, LOG_K_MAX)

            # Convert to concentration: exp(log_concentration) + min_concentration
            concentration = torch.exp(log_concentration) + self.min_concentration
        else:
            # For Gaussian mode, use softplus to ensure positive variance
            concentration = torch.nn.functional.softplus(concentration_embedding)
        
        return mean_embedding, concentration

    def encode_text_internal(
        self, text: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: Highly copy and pasted from the text forward function in CLIP
        concentration_embedding_batched = self.text_concentration_embedding.unsqueeze(0).expand(text.shape[0], -1, -1)
        x = torch.cat([
                self.model.token_embedding(text).float(), 
                concentration_embedding_batched
            ], dim=1
        )  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding.float()
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD


        x = self.model.ln_final(x).float()
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        mean_embedding = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.mean_text_projection
        # Concentration token is always at the last position (appended at the end)
        concentration_embedding = x[:, -1, :] @ self.var_text_projection

        if self.model_type == "Spherical":
            concentration_embedding = concentration_embedding.squeeze(-1)
            # Use log space + learnable scale: log_concentration = log_scale + projection_output
            # This works well with normalized features (projection learns relative differences)
            log_concentration_raw = concentration_embedding  # Already in log space from projection
            log_concentration = self.log_concentration_scale_text + log_concentration_raw

            # Clamp log_concentration to avoid overflow / NaNs in exp and downstream KL
            LOG_K_MIN = -5.0   # exp(-5)  ~ 0.0067
            LOG_K_MAX = 10.0   # exp(10)  ~ 22k
            log_concentration = torch.clamp(log_concentration, LOG_K_MIN, LOG_K_MAX)

            # Convert to concentration: exp(log_concentration) + min_concentration
            concentration = torch.exp(log_concentration) + self.min_concentration
        else:
            # For Gaussian mode, use softplus to ensure positive variance
            concentration = torch.nn.functional.softplus(concentration_embedding)
        
        return mean_embedding, concentration

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
            mean, var = self.encode_text_internal(text_tokens)
        else:
            with torch.no_grad():
                mean, var = self.encode_text_internal(text_tokens)

        return mean, var

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
