import os
import clip
import requests
import logging
from PIL import Image
from io import BytesIO
from datasets import load_dataset
import torch
import dotenv
import tqdm
from Datasets.preProcess import clip_preprocessor
import json

dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LaionDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for streaming the LAION dataset from Hugging Face Hub.

    This dataset streams data sample by sample, downloading images on-the-fly.
    It does not require downloading the entire dataset beforehand.
    """

    def __init__(self, data_dir: str = "Data", tokenize: bool = True, max_samples: int | None = None):
        super().__init__()
        laoin_path = os.path.join(data_dir, "laion")

        with open(os.path.join(laoin_path, "captions.json"), "r") as f:
            captions_data = json.load(f)

        self.data = []
        for filename in sorted(os.listdir(laoin_path)):
            if not filename.endswith('.jpg'):
                continue

            index = int(filename.split("_")[1].split(".")[0])
            caption = captions_data['captions'][index]
            self.data.append({
                "image_path": os.path.join(laoin_path, filename),
                "caption": caption,
            })

            if max_samples and len(self.data) >= max_samples:
                break

        self.tokenize = tokenize
        self.preprocess = clip_preprocessor()

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
    def collate_function(batch: list[tuple[torch.Tensor | None, torch.Tensor | None, str]]):
        # Text must be tokenized already
        images = torch.stack([img for img, _, _ in batch if img is not None])
        texts = torch.cat([text for _, text, _ in batch if text is not None])
        captions = [caption for _, _, caption in batch if caption is not None]

        return images, texts, captions
    
    @staticmethod
    def download(max_samples: int, data_dir: str = "Data"):
        dataset = load_dataset(
            "laion/relaion400m",
            split='train',
            streaming=True,
            token=os.environ.get("HUGGING_FACE_TOKEN"),
        )

        laoin_path = os.path.join(data_dir, "laion")
        if os.path.exists(laoin_path):
            logger.info("LAION dataset samples already downloaded.")
            return
        
        os.makedirs(laoin_path)
        logger.info("Downloading LAION dataset samples...")

        captions = []
        num_saved_images = 0
        with tqdm.tqdm(total=max_samples, desc="Downloading LAION samples") as pbar:
            for item in dataset:
                if num_saved_images >= max_samples:
                    break
                
                try:
                    # Download the image from the URL
                    response = requests.get(item["url"], timeout=10) # type: ignore
                    response.raise_for_status()

                    # Open image and convert to RGB
                    image = Image.open(BytesIO(response.content)).convert("RGB")

                    # Write image to disk
                    image_filename = os.path.join(laoin_path, f"image_{num_saved_images}.jpg")
                    image.save(image_filename)

                    # Save the caption, which later gets written to a JSON file
                    captions.append(item["caption"]) # type: ignore

                    # Register the saved image and caption
                    num_saved_images += 1
                    pbar.update()

                except (requests.RequestException, IOError, TypeError, ValueError) as e:
                    # If an image fails to download or process, log the error and skip it.
                    logger.warning(
                        f"Skipping item. Could not load image from URL {item.get('url', 'N/A')}. Reason: {e}") # type: ignore
                    
        with open(os.path.join(laoin_path, "captions.json"), "w") as f:
            json.dump({'captions': captions}, f)

   
