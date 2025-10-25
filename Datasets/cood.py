import zipfile
import clip
import torch
import os
import logging
from PIL import Image
from Datasets.download_from_google import download_from_google
from Datasets.preProcess import clip_preprocessor

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# https://arxiv.org/html/2407.05897v2#:~:text=The%20models%20trained%20on%20CommonPool,Report%20issue%20for%20preceding%20element


class CoodDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str = "Data", tokenize: bool = True, max_samples: int | None = None):
        self.data_dir = os.path.join(data_dir, 'cood', 'ImageNet-AO-filtered')
        self.tokenize = tokenize
        self.preprocess = clip_preprocessor()

        self.data = []
        for folder in sorted(os.listdir(self.data_dir)):
            caption = folder
            folder_path = os.path.join(self.data_dir, folder)

            for image in sorted(os.listdir(folder_path)):
                img_path = os.path.join(folder_path, image)
                if os.path.isfile(img_path):
                    self.data.append({
                        "image_path": img_path,
                        "caption": caption,
                    })
                if max_samples and len(self.data) >= max_samples:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image_path = self.data[index]["image_path"]
        caption = self.data[index]["caption"]

        image = Image.open(image_path).convert('RGB')
        imageTensor = self.preprocess(image)
        
        if self.tokenize:
            text = clip.tokenize(caption, truncate=True)

        return imageTensor, text, caption

    @staticmethod
    def collate_function(batch: list[tuple[torch.Tensor, torch.Tensor, str]]):
        # Text must be tokenized already
        images = torch.stack([img for img, _, _ in batch])
        texts = torch.cat([text for _, text, _ in batch])
        captions = [caption for _, _, caption in batch]

        return images, texts, captions

    @staticmethod
    def download(data_dir: str = "Data"):
        cood_path = os.path.join(data_dir, "cood")
        if not os.path.exists(cood_path):
            _download_cood( data_dir)
        else:
            logger.info("COOD dataset already exists. Skipping download.")


def _download_cood(data_dir: str = "Data"):
    # Extract this from the shareable link
    coco_directory = os.path.join(data_dir, "cood")
    if not os.path.exists(coco_directory):
        os.makedirs(coco_directory)

    file_id = '1qSoz1xmu1kHoZSF2IYRvTD1DTNXNZhKw'
    output_filename = os.path.join(coco_directory, 'downloaded_file.zip')
    download_from_google(file_id, output_filename)

    with zipfile.ZipFile(output_filename, 'r') as zip_ref:
        zip_ref.extractall(coco_directory)
    os.remove(output_filename)

    logger.info("Cood downloaded successfully.")
