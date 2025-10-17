"""
CLIP embedding extraction functionality for von Mises-Fisher mixture modeling.
"""

import torch
import clip
import numpy as np
from PIL import Image
from typing import List, Union, Tuple
import os


class CLIPEmbeddingExtractor:
    """Extract CLIP embeddings from text and images."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize CLIP model.

        Args:
            model_name: CLIP model variant to use
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def extract_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Extract CLIP embeddings from text inputs.

        Args:
            texts: List of text strings

        Returns:
            Normalized embeddings as numpy array
        """
        with torch.no_grad():
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            # Normalize embeddings to unit sphere
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()

    def extract_image_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract CLIP embeddings from image files.

        Args:
            image_paths: List of paths to image files

        Returns:
            Normalized embeddings as numpy array
        """
        embeddings = []

        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                # Preprocess image using CLIP's preprocessing
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    # Normalize embeddings to unit sphere
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    embeddings.append(image_features.cpu().numpy())

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                # Use zero vector as fallback
                embeddings.append(
                    np.zeros((1, 512))
                )  # CLIP ViT-B/32 has 512-dim embeddings

        return np.vstack(embeddings)

    def extract_multimodal_embeddings(
        self, texts: List[str] = None, image_paths: List[str] = None
    ) -> np.ndarray:
        """
        Extract embeddings from both text and images.

        Args:
            texts: List of text strings
            image_paths: List of paths to image files

        Returns:
            Combined normalized embeddings as numpy array
        """
        all_embeddings = []

        if texts:
            text_embeddings = self.extract_text_embeddings(texts)
            all_embeddings.append(text_embeddings)

        if image_paths:
            image_embeddings = self.extract_image_embeddings(image_paths)
            all_embeddings.append(image_embeddings)

        if not all_embeddings:
            raise ValueError("Must provide either texts or image_paths")

        return np.vstack(all_embeddings)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of CLIP embeddings."""
        # CLIP ViT-B/32 has 512-dimensional embeddings
        return 512


def create_sample_data(
    extractor: CLIPEmbeddingExtractor, num_samples: int = 1000
) -> np.ndarray:
    """
    Create sample CLIP embeddings for testing the mixture model.

    Args:
        extractor: CLIPEmbeddingExtractor instance
        num_samples: Number of sample embeddings to generate

    Returns:
        Array of sample embeddings
    """
    # Create diverse text samples
    sample_texts = [
        "a photo of a cat",
        "a dog playing in the park",
        "a beautiful sunset over mountains",
        "a person reading a book",
        "a car driving on the highway",
        "a bird flying in the sky",
        "a house with a garden",
        "a computer on a desk",
        "a painting of flowers",
        "a child playing with toys",
    ] * (num_samples // 10 + 1)

    # Add some variation
    import random

    random.shuffle(sample_texts)
    sample_texts = sample_texts[:num_samples]

    return extractor.extract_text_embeddings(sample_texts)


if __name__ == "__main__":
    # Example usage
    extractor = CLIPEmbeddingExtractor()

    # Test with sample texts
    sample_texts = [
        "a red apple on a table",
        "a blue car in the street",
        "a green tree in the forest",
        "a yellow sun in the sky",
    ]

    embeddings = extractor.extract_text_embeddings(sample_texts)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {extractor.get_embedding_dimension()}")
    print(f"Sample embedding norm: {np.linalg.norm(embeddings[0])}")
