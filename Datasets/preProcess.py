from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# Stolen from clip implementation
_CLIP_IMAGE_SIZE = 224


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def clip_preprocessor():
    return Compose([
        Resize(_CLIP_IMAGE_SIZE, interpolation=BICUBIC),
        CenterCrop(_CLIP_IMAGE_SIZE),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
