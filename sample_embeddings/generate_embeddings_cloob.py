"""
Generate embeddings for different datasets using CLIP models.
Supports COCO, LAION sample, custom images, and text datasets.
"""

from cloob import model_pt, pretrained
import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import argparse
import logging

from datasetLoader import DatasetLoader

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_CLOOB_CONFIG = "cloob_laion_400m_vit_b_16_16_epochs"


class EmbeddingGeneratorCLOOB:
    """Generate CLIP embeddings for various datasets."""

    def __init__(
        self,
        config=DEFAULT_CLOOB_CONFIG,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: CLIP model name (e.g., "ViT-B/32", "ViT-L/14")
            pretrained_name: Pretrained checkpoint name (for OpenCLIP)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            use_openclip: Whether to use OpenCLIP instead of original CLIP
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.configName = config

        logger.info(f"Initializing cloob {self.configName} on {self.device}")

        config = pretrained.get_config(self.configName)
        self.model = model_pt.get_pt_model(config)
        checkpoint = pretrained.download_checkpoint(config)
        self.model.load_state_dict(model_pt.get_pt_params(config, checkpoint))
        self.model.eval().requires_grad_(False).to(self.device)

    def generate_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for text inputs.

        Args:
            texts: List of text strings

        Returns:
            Normalized embeddings as numpy array
        """
        embeddings = []
        batch_size = 32  # Process in batches to avoid memory issues

        logger.info(f"Generating embeddings for {len(texts)} texts...")

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing text batches"):
            batch_texts = texts[i : i + batch_size]

            # TODO: Truncate???
            text_tokens = self.model.tokenizer(batch_texts).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)

                # Normalize to unit sphere
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embeddings.append(text_features.cpu().numpy())

        return np.vstack(embeddings)

    def generate_image_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """
        Generate embeddings for image files.

        Args:
            image_paths: List of paths to image files

        Returns:
            Normalized embeddings as numpy array
        """
        embeddings = []
        batch_size = 16  # Smaller batch size for images

        logger.info(f"Generating embeddings for {len(image_paths)} images...")

        for i in tqdm(
            range(0, len(image_paths), batch_size), desc="Processing image batches"
        ):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []

            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = self.preprocess(image).unsqueeze(0)
                    batch_images.append(image_tensor)
                except Exception as e:
                    logger.warning(f"Error loading image {img_path}: {e}")
                    # Create a dummy black image as fallback
                    dummy_image = Image.new("RGB", (224, 224), color="black")
                    image_tensor = self.preprocess(dummy_image).unsqueeze(0)
                    batch_images.append(image_tensor)

            if batch_images:
                batch_tensor = torch.cat(batch_images, dim=0).to(self.device)

                with torch.no_grad():
                    image_features = self.model.encode_image(batch_tensor)

                    # Normalize to unit sphere
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                    embeddings.append(image_features.cpu().numpy())

        return np.vstack(embeddings) if embeddings else np.array([])


def save_embeddings(
    embeddings: np.ndarray,
    labels: List[str],
    types: List[str],
    metadata: Dict,
    cache_file: str,
):
    """
    Save embeddings to cache file.

    Args:
        embeddings: Embedding array
        labels: Labels for each embedding
        types: Types for each embedding ("text" or "image")
        metadata: Additional metadata
        cache_file: Output cache file path
    """
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    np.savez(
        cache_file, embeddings=embeddings, labels=labels, types=types, metadata=metadata
    )

    logger.info(f"Saved embeddings to {cache_file}")


def load_embeddings(cache_file: str) -> Tuple[np.ndarray, List[str], List[str], Dict]:
    """
    Load embeddings from cache file.

    Args:
        cache_file: Cache file path

    Returns:
        Tuple of (embeddings, labels, types, metadata)
    """
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")

    cache = np.load(cache_file, allow_pickle=True)
    embeddings = cache["embeddings"]
    labels = cache["labels"].tolist()
    types = cache["types"].tolist()
    metadata = cache["metadata"].item() if "metadata" in cache else {}

    logger.info(f"Loaded embeddings from {cache_file}")
    return embeddings, labels, types, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate CLOOB embeddings for various datasets"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["coco", "laion_sample", "custom_images", "custom_text"],
        default="coco",
        help="Dataset to use",
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--custom_path", type=str, help="Path to custom images dir or text file"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train2017",
        choices=["train2017", "val2017"],
        help="COCO dataset split",
    )
    parser.add_argument(
        "--max_samples", type=int, help="Maximum number of samples to process"
    )

    # Model configuration
    parser.add_argument(
        "--model_name", type=str, default="ViT-B/32", help="CLIP model name"
    )
    parser.add_argument(
        "--pretrained_name", type=str, help="Pretrained checkpoint name (for OpenCLIP)"
    )
    parser.add_argument(
        "--use_openclip",
        action="store_true",
        help="Use OpenCLIP instead of original CLIP",
    )
    parser.add_argument("--device", type=str, help="Device to use (cuda/cpu)")

    # Output configuration
    parser.add_argument("--cache_file", type=str, help="Output cache file path")
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recompute even if cache exists",
    )

    # Processing options
    parser.add_argument(
        "--include_images",
        action="store_true",
        default=True,
        help="Include image embeddings",
    )
    parser.add_argument(
        "--include_texts",
        action="store_true",
        default=True,
        help="Include text embeddings",
    )

    args = parser.parse_args()

    # Generate cache file name if not provided
    if not args.cache_file:
        model_name_safe = args.model_name.replace("/", "-")
        args.cache_file = (
            f"data/clip_embeddings_cache_{args.dataset}_{model_name_safe}.npz"
        )

    # Check if cache exists and we don't want to force recompute
    if os.path.exists(args.cache_file) and not args.force_recompute:
        logger.info(f"Cache file exists: {args.cache_file}")
        embeddings, labels, types, metadata = load_embeddings(args.cache_file)
        logger.info(f"Loaded {len(embeddings)} embeddings")
        return embeddings, labels, types, metadata

    # Initialize embedding generator
    generator = EmbeddingGeneratorCLOOB(
        model_name=args.model_name,
        pretrained_name=args.pretrained_name,
        device=args.device,
        use_openclip=args.use_openclip,
    )

    # Load dataset
    if args.dataset == "coco":
        data = DatasetLoader.load_coco_dataset(
            data_dir=args.data_dir, split=args.split, max_samples=args.max_samples
        )
    elif args.dataset == "laion_sample":
        data = DatasetLoader.load_laion_sample()
    elif args.dataset == "custom_images":
        if not args.custom_path:
            raise ValueError("--custom_path required for custom_images dataset")
        data = DatasetLoader.load_custom_images(args.custom_path, args.max_samples)
    elif args.dataset == "custom_text":
        if not args.custom_path:
            raise ValueError("--custom_path required for custom_text dataset")
        data = DatasetLoader.load_text_file(args.custom_path, args.max_samples)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Generate embeddings
    all_embeddings = []
    all_labels = []
    all_types = []

    # Extract texts and image paths
    texts = []
    image_paths = []
    labels = []

    for item in data:
        if args.include_texts and item["text"]:
            texts.append(item["text"])
            labels.append(
                str(
                    item.get(
                        "image_id",
                        item.get("text_id", item.get("sample_id", "unknown")),
                    )
                )
            )

        if args.include_images and item["image_path"]:
            image_paths.append(item["image_path"])
            if not args.include_texts:  # Only add labels if not already added for text
                labels.append(
                    str(
                        item.get(
                            "image_id",
                            item.get("text_id", item.get("sample_id", "unknown")),
                        )
                    )
                )

    # Generate text embeddings
    if texts:
        logger.info("Generating text embeddings...")
        text_embeddings = generator.generate_text_embeddings(texts)
        all_embeddings.append(text_embeddings)
        all_labels.extend(labels[: len(texts)])
        all_types.extend(["text"] * len(texts))

    # Generate image embeddings
    if image_paths:
        logger.info("Generating image embeddings...")
        image_embeddings = generator.generate_image_embeddings(image_paths)
        all_embeddings.append(image_embeddings)
        if args.include_texts:
            # If we already have text labels, we need to adjust
            all_labels.extend(labels[: len(image_paths)])
        else:
            all_labels.extend(labels)
        all_types.extend(["image"] * len(image_paths))

    # Combine all embeddings
    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings)
    else:
        raise ValueError(
            "No embeddings generated. Check your dataset and include options."
        )

    # Create metadata
    metadata = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "pretrained_name": args.pretrained_name,
        "use_openclip": args.use_openclip,
        "n_samples": len(final_embeddings),
        "embedding_dim": final_embeddings.shape[1],
        "include_images": args.include_images,
        "include_texts": args.include_texts,
    }

    # Save embeddings
    save_embeddings(final_embeddings, all_labels, all_types, metadata, args.cache_file)

    logger.info(
        f"Generated {len(final_embeddings)} embeddings with shape {final_embeddings.shape}"
    )
    logger.info(f"Types: {dict(zip(*np.unique(all_types, return_counts=True)))}")

    return final_embeddings, all_labels, all_types, metadata


if __name__ == "__main__":
    main()
