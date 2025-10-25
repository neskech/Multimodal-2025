"""
CLOOB embedding extraction functionality for von Mises-Fisher mixture modeling.
"""

import torch
import sys
import os
import ssl
import urllib.request
import clip
import types
from PIL import Image
from typing import List, Union
from Models.clipInterface import ClipInterface

# Add cloob-training to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..',
                             'cloob-training'))
from Datasets.preProcess import clip_preprocessor
from cloob_training import model_pt, pretrained


# Global constant for model configuration
MODEL_NAME = "cloob_laion_400m_vit_b_16_32_epochs"


class CLOOBModel(ClipInterface):
    """
    CLOOB embedding extractor as a PyTorch module.
    Hard-coded to use ViT-B/16 architecture.
    """

    def __init__(self, device: str | None = None):
        """
        Initialize CLOOB model.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = MODEL_NAME

        # Disable SSL verification to handle expired certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        urllib.request.install_opener(
            urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl_context)
            )
        )

        # Load CLOOB model
        self.config = pretrained.get_config(MODEL_NAME)
        self.model = model_pt.get_pt_model(self.config)
        checkpoint = pretrained.download_checkpoint(self.config)
        self.model.load_state_dict(model_pt.get_pt_params(self.config, checkpoint))
        self.model = self.model.to(self.device)

    def freeze_for_finetuning(self):
        """Freeze CLOOB model parameters for finetuning."""
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.image_encoder.proj.requires_grad = True
        self.model.text_encoder.proj.requires_grad = True

        def text_forward(instance, x):
            eot_mask = x == instance.eot_token
            padding_mask = torch.cumsum(eot_mask, dim=-1) == 0 | eot_mask
            x = instance.embed(x)
            x = instance.pos_embed(x)
            for layer in instance.layers:
                x = layer(x, padding_mask)
            x.requires_grad_(True) # NOTE: New line
            x = x[:, 0]
            x = instance.proj(x)
            x = torch.nn.functional.normalize(x, dim=-1)
            return x

        self.model.text_encoder.forward = types.MethodType(text_forward, self.model.text_encoder)

        def visual_forward(instance, x):
            x = instance.embed(x)
            x = x.reshape([x.shape[0], x.shape[1], -1]).permute([0, 2, 1])
            x = torch.cat([instance.class_embed[None, None].repeat([x.shape[0], 1, 1]), x], dim=1)
            x = instance.pos_embed(x)
            for layer in instance.layers:
                x = layer(x)
            x.requires_grad_(True) # NOTE: New line
            x = x[:, 0]
            x = instance.proj(x)
            x = torch.nn.functional.normalize(x, dim=-1)
            return x
        
        self.model.image_encoder.forward = types.MethodType(visual_forward, self.model.image_encoder)


    def get_config(self):
        return self.config
 
    def encode_image_tensors(self,
                             image_tensors: torch.Tensor,
                             requires_grad: bool = True) -> torch.Tensor:
        """
        Encode image tensors to embeddings.

        Args:
            image_tensors: Batch of image tensors
                          Shape: [batch_size, 3, 224, 224]
            requires_grad: Whether to compute gradients (default: True)

        Returns:
            Image embeddings
            Shape: [batch_size, 512]
        """
        if requires_grad:
            image_features = self.model.image_encoder(image_tensors)
        else:
            with torch.no_grad():
                image_features = self.model.image_encoder(image_tensors)

        return image_features

    def encode_text_tokens(
        self,
        text_tokens: torch.Tensor,
        requires_grad: bool = True
    ) -> torch.Tensor:
        """
        Encode tokenized text to embeddings.

        Args:
            text_tokens: Batch of tokenized text tensors
                        Shape: [batch_size, 77] (context_length)
            requires_grad: Whether to compute gradients (default: True)

        Returns:
            Text embeddings
            Shape: [batch_size, 512]
        """
        if requires_grad:
            text_features = self.model.text_encoder.forward(text_tokens)
        else:
            with torch.no_grad():
                text_features = self.model.text_encoder.forward(text_tokens)

        return text_features

    def encode_text(
        self,
        texts: Union[str, List[str]],
        requires_grad: bool = True,
    ) -> torch.Tensor:
        """
        Encode text strings to embeddings.

        Args:
            texts: Single text string or list of text strings
            requires_grad: Whether to compute gradients (default: True)

        Returns:
            Text embeddings
            Shape: [batch_size, 512]
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize text
        text_tokens = clip.tokenize(texts, truncate=True)
        text_tokens = text_tokens.to(self.device)

        return self.encode_text_tokens(
            text_tokens, requires_grad=requires_grad
        )

    def encode_images(
        self,
        image_paths: Union[str, List[str]],
        requires_grad: bool = True,
    ) -> torch.Tensor:
        """
        Encode images from file paths to embeddings.

        Args:
            image_paths: Single image path or list of image paths
            requires_grad: Whether to compute gradients (default: True)

        Returns:
            Image embeddings
            Shape: [batch_size, 512]
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
            image_tensors, requires_grad=requires_grad
        )

    def get_embedding_dimension(self) -> int:
        """Get the dimension of CLOOB embeddings."""
        # CLOOB ViT-B/16 has 512-dimensional embeddings
        return 512  # TODO: Check if this is correct
