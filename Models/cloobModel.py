"""
CLOOB embedding extraction functionality for von Mises-Fisher mixture modeling.
"""

import torch
import sys
import os
import ssl
import urllib.request
import clip
from PIL import Image
from typing import List, Union
from clipInterface import ClipInterface

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
        self.model.to(self.device)

    def get_config(self):
        return self.config
 
    def encode_image_tensors(self,
                             image_tensors: torch.Tensor,
                             requires_grad: bool = True,
                             normalize: bool = False) -> torch.Tensor:
        """
        Encode image tensors to embeddings.

        Args:
            image_tensors: Batch of image tensors
                          Shape: [batch_size, 3, 224, 224]
            requires_grad: Whether to compute gradients (default: True)
            normalize: Whether to normalize embeddings to unit sphere (default: False)

        Returns:
            Image embeddings
            Shape: [batch_size, 512]
        """
        if requires_grad:
            image_features = self.model.image_encoder(image_tensors)
        else:
            with torch.no_grad():
                image_features = self.model.image_encoder(image_tensors)

        if normalize:
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return image_features

    def encode_text_tokens(
        self,
        text_tokens: torch.Tensor,
        requires_grad: bool = True,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Encode tokenized text to embeddings.

        Args:
            text_tokens: Batch of tokenized text tensors
                        Shape: [batch_size, 77] (context_length)
            requires_grad: Whether to compute gradients (default: True)
            normalize: Whether to normalize embeddings to unit sphere (default: False)

        Returns:
            Text embeddings
            Shape: [batch_size, 512]
        """
        if requires_grad:
            text_features = self.model.text_encoder(text_tokens)
        else:
            with torch.no_grad():
                text_features = self.model.text_encoder(text_tokens)

        if normalize:
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return text_features

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
    ) -> torch.Tensor:
        """
        Encode images from file paths to embeddings.

        Args:
            image_paths: Single image path or list of image paths
            requires_grad: Whether to compute gradients (default: True)
            normalize: Whether to normalize embeddings to unit sphere (default: False)

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
            image_tensors, requires_grad=requires_grad, normalize=normalize
        )

    def get_embedding_dimension(self) -> int:
        """Get the dimension of CLOOB embeddings."""
        # CLOOB ViT-B/16 has 512-dimensional embeddings
        return 512  # TODO: Check if this is correct
