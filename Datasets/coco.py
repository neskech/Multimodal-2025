import json
import os
from typing import List, Dict, Optional, Literal, Union
import logging
from PIL import Image
import clip
import torch

from Datasets.preProcess import clip_preprocessor

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Type alias for compatibility with older Python versions
CocoSplit = Union[
    Literal["train2017"],
    Literal["val2017"],
    Literal["test2017"],
    Literal["unlabeled2017"],
]


class CocoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str = "Data",
        split: CocoSplit = "train2017",
        tokenize: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        self.tokenize = tokenize
        self.preprocess = clip_preprocessor()

        logger.info(f"Loading COCO {split} dataset...")
        coco_dir = os.path.join(self.data_dir, "coco")
        images_dir = os.path.join(coco_dir, "images", self.split)
        annotations_file = os.path.join(
            coco_dir, "annotations", f"captions_{self.split}.json"
        )

        with open(annotations_file, "r") as f:
            coco_data = json.load(f)

        # Create image ID to filename mapping
        image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}

        self.data = []
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]

            if image_id in image_id_to_file:
                img_file = image_id_to_file[image_id]
                img_path = os.path.join(images_dir, img_file)

                if os.path.exists(img_path):
                    self.data.append(
                        {
                            "image_path": img_path,
                            "caption": caption,
                        }
                    )

                    if self.max_samples and len(self.data) >= self.max_samples:
                        break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image_path = self.data[index]["image_path"]
        caption = self.data[index]["caption"]

        image = Image.open(image_path).convert("RGB")
        imageTensor = self.preprocess(image)

        if self.tokenize:
            caption = clip.tokenize(caption, truncate=True)

        return imageTensor, caption

    @staticmethod
    def collate_function(batch: List[tuple[torch.Tensor, torch.Tensor]]):
        # Text must be tokenized already
        images = torch.stack([img for img, _ in batch])
        captions = torch.cat([caption for _, caption in batch])
        return images, captions

    @staticmethod
    def download(
        download_script_path: str = "./download_coco.sh", data_dir: str = "Data"
    ):
        """
        Download COCO dataset if not already present.

        Args:
            data_dir: Directory to download COCO dataset into
        """
        import subprocess

        coco_dir = os.path.join(data_dir, "coco")
        if not os.path.exists(coco_dir):
            os.makedirs(coco_dir, exist_ok=True)
            logger.info("Downloading COCO dataset...")
            subprocess.run(["bash", download_script_path], check=True)
            logger.info("COCO dataset downloaded.")
        else:
            logger.info("COCO dataset already exists. Skipping download.")
