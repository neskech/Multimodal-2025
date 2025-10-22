from io import BytesIO
import zipfile
import clip
import requests
import torch
import os
import logging
from PIL import Image
from Datasets.download_from_google import download_from_google
from Datasets.preProcess import clip_preprocessor
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up logging


class CC12mDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str = 'Data', tokenize: bool = True, max_samples: int | None = None):
        super().__init__()
        self.cc12m_dir = os.path.join(data_dir, 'cc12m')
        self.tokenize = tokenize
        self.preprocess = clip_preprocessor()

        tsv_path = os.path.join(self.cc12m_dir, 'data.tsv')
        self.data = pd.read_csv(tsv_path, sep='\t', header=None, names=['url', 'caption'])
        if max_samples:
            self.data = self.data.head(max_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        try:
            # Download the image from the URL
            response = requests.get(self.data[index]["url"], timeout=10)
            response.raise_for_status()

            # Open image and convert to RGB
            image = Image.open(BytesIO(response.content)).convert("RGB")

            # Apply transformations to the image
            processed_image = self.preprocess(image)

            # The text is already available
            caption = self.data[index]["caption"]

            if self.tokenize:
                caption = clip.tokenize(caption, truncate=True)[0]

            return processed_image, caption

        except (requests.RequestException, IOError, TypeError, ValueError) as e:
            # If an image fails to download or process, log the error and skip it.
            logger.warning(
                f"Skipping item. Could not load image from URL {self.data[index]["url"]}. Reason: {e}")
            return None, None

    @staticmethod
    def collate_function(batch: list[tuple[torch.Tensor | None, torch.Tensor | None]]):
        # Text must be tokenized already
        images = torch.stack([img for img, _ in batch if img is not None])
        captions = torch.stack([caption for _, caption in batch if caption is not None])
        return images, captions

    @staticmethod
    def download(data_dir: str = "Data"):
        cc12m_path = os.path.join(data_dir, "cc12m")
        if not os.path.exists(cc12m_path):
            _download_cc12m()
        else:
            logger.info("CC12m dataset already exists. Skipping download.")


def _download_cc12m(data_dir: str = "Data"):
    # Extract this from the shareable link
    cc12m_directory = os.path.join(data_dir, "cc12m")
    if not os.path.exists(cc12m_directory):
        os.makedirs(cc12m_directory)

    file_id = '1mZ_sHAp7jpMfFVY2TFN9wZioYujoYfCL'
    output_filename = os.path.join(cc12m_directory, 'data.tsv')
    download_from_google(file_id, output_filename)

    logger.info("CC12m TSV file downloaded successfully.")
