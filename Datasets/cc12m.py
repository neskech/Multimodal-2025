from io import BytesIO
import zipfile
import clip
import requests
import torch
import os
import logging
from PIL import Image
import tqdm
from Datasets.download_from_google import download_from_google
from Datasets.preProcess import clip_preprocessor
import pandas as pd
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set up logging


class CC12mDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str = "Data",
        tokenize: bool = True,
        max_samples: int | None = None,
    ):
        super().__init__()
        self.cc12m_dir = os.path.join(data_dir, "cc12m")

        with open(os.path.join(self.cc12m_dir, "captions.json"), "r") as f:
            captions_data = json.load(f)

        self.data = []
        for filename in sorted(os.listdir(self.cc12m_dir)):
            if not filename.endswith(".jpg"):
                continue

            index = int(filename.split("_")[1].split(".")[0])
            caption = captions_data["captions"][index]
            self.data.append(
                {
                    "image_path": os.path.join(self.cc12m_dir, filename),
                    "caption": caption,
                }
            )

            if max_samples and len(self.data) >= max_samples:
                break

        self.tokenize = tokenize
        self.preprocess = clip_preprocessor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        try:
            # Download the image from the URL
            # print(self.data[index]["image_path"])
            # response = requests.get(self.data[index]["image_path"], timeout=10)
            # response.raise_for_status()

            # Open image and convert to RGB
            image = Image.open(self.data[index]["image_path"]).convert("RGB")

            # Apply transformations to the image
            processed_image = self.preprocess(image)

            # The text is already available
            caption = self.data[index]["caption"]

            if self.tokenize:
                caption = clip.tokenize(caption, truncate=True)

            return processed_image, caption

        except (requests.RequestException, IOError, TypeError, ValueError) as e:
            # If an image fails to download or process, log the error and skip it.
            logger.warning(
                "Skipping item. Could not load image from URL "
                + self.data[index]["image_path"]
                + ". Reason: {e}"
            )
            return None, None

    @staticmethod
    def collate_function(batch: list[tuple[torch.Tensor | None, torch.Tensor | None]]):
        # Text must be tokenized already
        images = torch.stack([img for img, _ in batch if img is not None])
        captions = torch.cat([caption for _, caption in batch if caption is not None])
        return images, captions

    @staticmethod
    def download(max_samples: int, data_dir: str = "Data"):
        cc12m_path = os.path.join(data_dir, "cc12m")

        if os.path.exists(cc12m_path):
            logger.info("CC12m dataset already exists. Skipping download.")
            return

        _download_cc12m()
        tsv_path = os.path.join(cc12m_path, "data.tsv")
        data = pd.read_csv(tsv_path, sep="\t", header=None, names=["url", "caption"])
        data = data.head(max_samples)

        captions = []
        num_saved_images = 0
        with tqdm.tqdm(total=max_samples, desc="Downloading CC12m samples") as pbar:
            for _, row in data.iterrows():
                if num_saved_images >= max_samples:
                    break

                url = row["url"]
                caption = row["caption"]
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()

                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    image_filename = os.path.join(
                        cc12m_path, f"image_{num_saved_images}.jpg"
                    )
                    image.save(image_filename)
                    captions.append(caption)

                    num_saved_images += 1
                    pbar.update()

                except (requests.RequestException, IOError, TypeError, ValueError) as e:
                    logger.warning(
                        f"Skipping item. Could not load image from URL {url}. Reason: {e}"
                    )

            with open(os.path.join(cc12m_path, "captions.json"), "w") as f:
                json.dump({"captions": captions}, f)


def _download_cc12m(data_dir: str = "Data"):
    # Extract this from the shareable link
    cc12m_directory = os.path.join(data_dir, "cc12m")
    if not os.path.exists(cc12m_directory):
        os.makedirs(cc12m_directory)

    file_id = "1mZ_sHAp7jpMfFVY2TFN9wZioYujoYfCL"
    output_filename = os.path.join(cc12m_directory, "data.tsv")
    download_from_google(file_id, output_filename)

    logger.info("CC12m TSV file downloaded successfully.")
