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
import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url

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
        dataset=None,
    ):
        super().__init__()
        print("cc12m dataset", dataset)
        if dataset is not None:
            self.wds_dataset = dataset
        else:
            # Load dataset from HuggingFace using webdataset
            fs = HfFileSystem()
            files = [
                fs.resolve_path(path)
                for path in fs.glob("hf://datasets/pixparse/cc12m-wds/*/-train-*.tar")
            ]
            urls = [
                hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset")
                for file in files
            ]
            urls = f"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(urls)}"

            self.wds_dataset = wds.WebDataset(urls).decode().shuffle(max_samples // 100)

        # Convert to list for indexing (if max_samples is specified)
        self.data = []
        for i, sample in enumerate(self.wds_dataset):
            if max_samples and i >= max_samples:
                break
            self.data.append(sample)

        self.tokenize = tokenize
        self.preprocess = clip_preprocessor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        try:
            sample = self.data[index]

            # Extract image and caption from webdataset sample
            # webdataset typically has keys like 'jpg', 'png' for images and 'txt', 'json' for text
            image = sample.get("jpg") or sample.get("png") or sample.get("image")
            caption = sample.get("txt") or sample.get("caption") or sample.get("text")

            if image is None or caption is None:
                logger.warning(
                    f"Skipping item at index {index}. Missing image or caption."
                )
                return None, None

            # If image is bytes, convert to PIL Image
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image)).convert("RGB")
            elif not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")

            # Apply transformations to the image
            processed_image = self.preprocess(image)

            # Process caption
            if isinstance(caption, bytes):
                caption = caption.decode("utf-8")

            if self.tokenize:
                caption = clip.tokenize(caption, truncate=True)

            return processed_image, caption

        except (IOError, TypeError, ValueError, KeyError) as e:
            logger.warning(f"Skipping item at index {index}. Reason: {e}")
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
