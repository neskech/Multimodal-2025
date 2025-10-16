import os
import clip
import requests
import logging
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from torch.utils.data import IterableDataset
import dotenv
from Datasets.preProcess import clip_preprocessor

dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamingLaionDataset(IterableDataset):
    """
    A PyTorch IterableDataset for streaming the LAION dataset from Hugging Face Hub.

    This dataset streams data sample by sample, downloading images on-the-fly.
    It does not require downloading the entire dataset beforehand.
    """

    def __init__(self, tokenize: bool = True, max_samples: int | None = None):
        super().__init__()
        logger.info("Loading LAION dataset in streaming mode")
        # Load the dataset in streaming mode
        self.dataset = load_dataset(
            "laion/relaion400m",
            split='train',
            streaming=True,
            token=os.environ.get("HUGGING_FACE_TOKEN"),
        )
        self.length = max_samples if max_samples is not None else len(
            self.dataset)
        self.tokenize = tokenize
        self.preprocess = clip_preprocessor()

    def __len__(self):
        return self.length

    def __iter__(self):
        """
        The core of the IterableDataset. This method yields processed samples.
        """
        for item in self.dataset:
            try:
                # Download the image from the URL
                response = requests.get(item["url"], timeout=10)
                response.raise_for_status()

                # Open image and convert to RGB
                image = Image.open(BytesIO(response.content)).convert("RGB")

                # Apply transformations to the image
                processed_image = self.preprocess(image)

                # The text is already available
                caption = item["caption"]

                if self.tokenize:
                    caption = clip.tokenize(caption, truncate=True)

                yield processed_image, caption

            except (requests.RequestException, IOError, TypeError, ValueError) as e:
                # If an image fails to download or process, log the error and skip it.
                logger.warning(
                    f"Skipping item. Could not load image from URL {item.get('url', 'N/A')}. Reason: {e}")
                continue
