"""
CLIP embedding extraction functionality for von Mises-Fisher mixture modeling.
"""

import torch
import clip
from torch import nn
from PIL import Image
from typing import List, Union
from AlignCLIP.align_clip.model import CLIP
from AlignCLIP.align_clip.factory import create_model_and_transforms
from huggingface_hub import hf_hub_download

# Global constant for model configuration
MODEL_NAME = "hf-hub:sarahESL/AlignCLIP"



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
        self.device = device or ("cuda"
                                 if torch.cuda.is_available() else "cpu")
        self.model_name = MODEL_NAME
        checkpoint_path = hf_hub_download(
            repo_id="sarahESL/AlignCLIP",
            filename="alignCLIP.pt"
        )
        # Load CLIP model
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name="ViT-L-16",  # Replace with actual architecture
            pretrained=checkpoint_path,
            device=device  # or "cpu"
        )
        self.model = model
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val

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
            image_features = self.model.encode_image(image_tensors)
        else:
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensors)

        if normalize:
            image_features = image_features / image_features.norm(dim=1,
                                                                  keepdim=True)

        return image_features

    def encode_text_tokens(self,
                           text_tokens: torch.Tensor,
                           requires_grad: bool = True,
                           normalize: bool = False) -> torch.Tensor:
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
            text_features = self.model.encode_text(text_tokens)
        else:
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)

        if normalize:
            text_features = text_features / text_features.norm(dim=1,
                                                               keepdim=True)
        return text_features

    def forward(
        self,
        image_tensors: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(image_tensors, text_tokens)

    def encode_text(self,
                    texts: Union[str, List[str]],
                    requires_grad: bool = True,
                    normalize: bool = False) -> torch.Tensor:
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

        return self.encode_text_tokens(text_tokens,
                                       requires_grad=requires_grad,
                                       normalize=normalize)

    def encode_images(self,
                      image_paths: Union[str, List[str]],
                      requires_grad: bool = True,
                      normalize: bool = False) -> torch.Tensor:
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
                image = Image.open(image_path).convert('RGB')
                # Use CLIP's preprocessing
                image_tensor = self.preprocess(image).unsqueeze(0)
                image_tensors.append(image_tensor)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Return a black image as fallback
                image_tensors.append(torch.zeros(1, 3, 224, 224))

        # Concatenate all images
        image_tensors = torch.cat(image_tensors, dim=0).to(self.device)

        return self.encode_image_tensors(image_tensors,
                                         requires_grad=requires_grad,
                                         normalize=normalize)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of CLIP embeddings."""
        # CLIP ViT-B/32 has 512-dimensional embeddings
        return 512
