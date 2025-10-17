import json
import os
from typing import List, Dict, Optional
import logging


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load different types of datasets."""

    @staticmethod
    def load_coco_dataset(
        data_dir: str = "data",
        split: str = "train2017",
        max_samples: Optional[int] = None,
    ) -> List[Dict]:
        """
        Load COCO dataset.

        Args:
            data_dir: Data directory containing COCO files
            split: Dataset split ("train2017" or "val2017")
            max_samples: Maximum number of samples to load

        Returns:
            List of data samples with image paths and captions
        """
        coco_dir = os.path.join(data_dir, "coco")
        images_dir = os.path.join(coco_dir, "images", split)
        annotations_file = os.path.join(
            coco_dir, "annotations", f"captions_{split}.json"
        )

        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"COCO annotations not found: {annotations_file}")

        logger.info(f"Loading COCO {split} dataset...")

        with open(annotations_file, "r") as f:
            coco_data = json.load(f)

        # Create image ID to filename mapping
        image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}

        data = []
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]

            if image_id in image_id_to_file:
                img_file = image_id_to_file[image_id]
                img_path = os.path.join(images_dir, img_file)

                if os.path.exists(img_path):
                    data.append(
                        {
                            "image_path": img_path,
                            "text": caption,
                            "image_id": image_id,
                            "dataset": "coco",
                        }
                    )

                    if max_samples and len(data) >= max_samples:
                        break

        logger.info(f"Loaded {len(data)} COCO samples")
        return data

    @staticmethod
    def load_cood_dataset(
        data_dir: str = "data", split: str = "train", max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Load COOD dataset.

        Args:
            data_dir: Data directory containing COOD files
            split: Dataset split ("train" or "val")
            max_samples: Maximum number of samples to load
        Returns:
            List of data samples with image paths and captions
        """
        return

    @staticmethod
    def load_laion_sample() -> List[Dict]:
        """
        Load a sample of LAION-style data (text-only for demonstration).

        Returns:
            List of text samples
        """
        texts = [
            "A photo of a cat sitting on a keyboard",
            "A majestic mountain range at sunset, painted by a professional artist",
            "An abstract sculpture made of glass and metal",
            "A plate of sushi with chopsticks on a wooden table",
            "A wide shot of a bustling city street at night with neon lights",
            "A close-up of a human eye with detailed iris patterns",
            "A watercolor painting of a sunflower in a field",
            "A digital art piece of a futuristic spaceship in space",
            "A dog wearing sunglasses and a funny hat",
            "A vintage car parked in front of a retro diner",
            "A chef preparing pasta in a professional kitchen",
            "A library with tall bookshelves and reading tables",
            "A concert hall with musicians performing on stage",
            "A beach with crystal clear water and white sand",
            "A forest path covered with autumn leaves",
            "A modern skyscraper with glass facade reflecting clouds",
            "A child playing with colorful building blocks",
            "A garden full of blooming roses and butterflies",
            "A train crossing a bridge over a river",
            "A cozy coffee shop with customers reading books",
        ]

        data = []
        for i, text in enumerate(texts):
            data.append(
                {
                    "text": text,
                    "image_path": None,
                    "sample_id": i,
                    "dataset": "laion_sample",
                }
            )

        logger.info(f"Loaded {len(data)} LAION sample texts")
        return data

    @staticmethod
    def load_custom_images(
        image_dir: str, max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Load custom images from a directory.

        Args:
            image_dir: Directory containing images
            max_samples: Maximum number of images to load

        Returns:
            List of image data
        """
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_paths = []

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_paths.append(os.path.join(root, file))

                    if max_samples and len(image_paths) >= max_samples:
                        break
            if max_samples and len(image_paths) >= max_samples:
                break

        data = []
        for i, img_path in enumerate(image_paths):
            # Use filename (without extension) as label
            label = os.path.splitext(os.path.basename(img_path))[0]
            data.append(
                {
                    "image_path": img_path,
                    "text": label.replace("_", " "),  # Simple text from filename
                    "image_id": i,
                    "dataset": "custom_images",
                }
            )

        logger.info(f"Loaded {len(data)} custom images")
        return data

    @staticmethod
    def load_text_file(text_file: str, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load text data from a file (one text per line).

        Args:
            text_file: Path to text file
            max_samples: Maximum number of texts to load

        Returns:
            List of text data
        """
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Text file not found: {text_file}")

        with open(text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        if max_samples:
            texts = texts[:max_samples]

        data = []
        for i, text in enumerate(texts):
            data.append(
                {
                    "text": text,
                    "image_path": None,
                    "text_id": i,
                    "dataset": "custom_text",
                }
            )

        logger.info(f"Loaded {len(data)} texts")
        return data
