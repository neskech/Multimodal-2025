from typing import List, Tuple
import torch
from transformers import GPT2Tokenizer


class CaptioningMetric:
    """
    Metric to compute BLEU score for captioning tasks.
    """

    def __init__(self):
        pass

    def compute(
        self,
        embeddings: List[Tuple[torch.Tensor, torch.Tensor]],
        captions: List[str],
        clip_model,
        data_dir: str = "Data",
    ) -> float:
        """
        1. Finetune a captioning model on the provided embeddings and captions.
        2. Evaluate the model using BLEU score.
        """
        from Models.clipCaptionModel import ClipCaptionModel
        from transformers import GPT2Tokenizer

        # Initialize tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Train the captioning model on CC12m dataset
        trained_model, _, _, val_embeddings, val_captions = (
            train_caption_model_on_cc12m(
                clip_model=clip_model, num_epochs=3, batch_size=4, data_dir=data_dir
            )
        )

        # Evaluate the trained model on validation set from CC12m
        predictions = []
        for img_emb in val_embeddings:
            img_emb = img_emb.unsqueeze(0).to(device)
            generated_caption = generate_caption(trained_model, img_emb, tokenizer)
            predictions.append(generated_caption)

        return bleu_score(val_captions, predictions=predictions)


def bleu_score(predictions, references):
    """
    Compute a simple BLEU score for a list of predictions and references.

    Args:
        predictions (List[str]): List of predicted sentences.
        references (List[str]): List of reference sentences.

    Returns:
        float: Average BLEU score across all predictions.
    """
    from nltk.translate.bleu_score import sentence_bleu

    total_score = 0.0
    for pred, ref in zip(predictions, references):
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        score = sentence_bleu(ref_tokens, pred_tokens)
        total_score += score

    return total_score / len(predictions) if predictions else 0.0


# Import necessary libraries for captioning model training
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import json
from datasetLoader import DatasetLoader
from Models.clipCaptionModel import ClipCaptionModel
from typing import List, Tuple, Optional
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CaptionTrainingDataset(Dataset):
    """Dataset for training CLIP captioning model using image embeddings."""

    def __init__(self, image_embeddings, captions, tokenizer, max_length=77):
        """
        Args:
            image_embeddings: Tensor of pre-computed image embeddings (N, D)
            captions: List of caption strings
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.image_embeddings = image_embeddings
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_embedding = self.image_embeddings[idx]
        caption = self.captions[idx]

        # Tokenize caption
        tokens = self.tokenizer.encode(
            caption, max_length=self.max_length, truncation=True, padding="max_length"
        )
        tokens = torch.tensor(tokens, dtype=torch.long)

        return {
            "image_embedding": image_embedding,
            "tokens": tokens,
            "caption": caption,
        }


def evaluate_captioning_model(
    caption_model, clip_model, data_samples, tokenizer, max_samples=50
):
    """
    Evaluate the captioning model using BLEU score.

    Args:
        caption_model: Trained captioning model
        clip_model: CLIP model for generating image embeddings
        data_samples: List of data samples for evaluation
        tokenizer: GPT-2 tokenizer
        max_samples: Maximum number of samples to evaluate

    Returns:
        Average BLEU score
    """
    caption_model.eval()

    # Limit samples for faster evaluation
    eval_samples = data_samples[:max_samples]

    predictions = []
    references = []

    with torch.no_grad():
        for sample in tqdm(eval_samples, desc="Evaluating"):
            # Generate image embedding
            image_embedding = (
                clip_model.encode_images([sample["image_path"]])[0]
                .unsqueeze(0)
                .to(device)
            )

            # Generate caption
            generated_caption = generate_caption(
                caption_model, image_embedding, tokenizer
            )

            predictions.append(generated_caption)
            references.append(sample["text"])

    # Calculate BLEU score using the function defined earlier
    bleu_score_result = bleu_score(predictions, references)
    return bleu_score_result


def generate_caption(
    caption_model, image_embedding, tokenizer, max_length=50, temperature=0.8
):
    """
    Generate a caption for a given image embedding.

    Args:
        caption_model: Trained captioning model
        image_embedding: Image embedding from CLIP model
        tokenizer: GPT-2 tokenizer
        max_length: Maximum caption length
        temperature: Sampling temperature

    Returns:
        Generated caption as string
    """
    caption_model.eval()

    with torch.no_grad():
        # Start with just the image embedding
        generated_ids = []

        # Get dummy tokens for prefix
        dummy_tokens = caption_model.get_dummy_token(1, device)

        for _ in range(max_length):
            # Prepare input tokens
            if len(generated_ids) == 0:
                input_tokens = dummy_tokens
            else:
                input_tokens = torch.tensor(
                    [generated_ids], dtype=torch.long, device=device
                )

            # Forward pass
            outputs = caption_model(input_tokens, image_embedding)
            logits = outputs.logits

            # Get next token probabilities
            next_token_logits = logits[0, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(next_token_probs, 1).item()

            # Check for end token
            if next_token == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)

        # Decode the generated caption
        caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return caption


def train_caption_model(image_embeddings, captions, num_epochs=5, batch_size=8):
    """
    Train a captioning model on the provided image embeddings and captions.

    Args:
        image_embeddings: Tensor of pre-computed image embeddings (N, D)
        captions: List of caption strings
        num_epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Tuple of (trained_model, training_losses, bleu_scores)
    """
    from transformers import GPT2Tokenizer

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    caption_model = ClipCaptionModel(prefix_length=10, prefix_size=512, clip_length=512)
    caption_model.to(device)

    # Prepare dataset and dataloader
    dataset = CaptionTrainingDataset(image_embeddings, captions, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(caption_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    training_losses = []
    bleu_scores = []

    for epoch in range(num_epochs):
        caption_model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            image_embeddings_batch = batch["image_embedding"].to(device)
            tokens = batch["tokens"].to(device)

            optimizer.zero_grad()
            outputs = caption_model(tokens[:, :-1], image_embeddings_batch)
            logits = outputs.logits

            loss = criterion(
                logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        training_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}")

        # Note: BLEU score evaluation would require validation embeddings and captions
        # For now, we'll skip this during training
        bleu_scores.append(0.0)

    return caption_model, training_losses, bleu_scores


def train_caption_model_on_cc12m(
    clip_model,
    data_dir="Data",
    max_samples=1000,
    batch_size=8,
    num_epochs=5,
):
    """
    Complete pipeline to train a captioning model on CC12m (Conceptual Captions) dataset.
    Uses DataLoader to efficiently process the dataset similar to finetune.ipynb.

    Args:
        clip_model: Pre-trained CLIP model
        data_dir: Directory containing CC12m dataset
        max_samples: Maximum number of training samples
        batch_size: Training batch size
        num_epochs: Number of training epochs

    Returns:
        Tuple of (trained_model, training_losses, bleu_scores, val_embeddings, val_captions)
    """
    from Datasets.cc12m import CC12mDataset

    # Download and load CC12m dataset
    print("Loading CC12m dataset...")
    CC12mDataset.download(max_samples=max_samples, data_dir=data_dir)

    # Load the full dataset (tokenize=False since we don't need CLIP tokens, just captions)
    all_data = CC12mDataset(data_dir=data_dir, tokenize=False, max_samples=max_samples)

    # Split into train and validation
    num_train = int(0.8 * len(all_data))
    train_dataset = torch.utils.data.Subset(all_data, range(0, num_train))
    val_dataset = torch.utils.data.Subset(all_data, range(num_train, len(all_data)))

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create a custom collate function that filters out None values
    def collate_fn_filter(batch):
        # Filter out None values
        batch = [
            (img, cap) for img, cap in batch if img is not None and cap is not None
        ]
        if len(batch) == 0:
            return None, None
        images = torch.stack([img for img, _ in batch])
        captions = [cap for _, cap in batch]
        return images, captions

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,  # Larger batch for embedding computation
        shuffle=False,
        collate_fn=collate_fn_filter,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_filter
    )

    # Pre-compute embeddings for all training data
    print("Pre-computing image embeddings for training set...")
    train_embeddings = []
    train_captions = []

    with torch.no_grad():
        for images, captions in tqdm(train_loader, desc="Computing train embeddings"):
            if images is None:
                continue
            # Get image embeddings from CLIP model
            embeddings = clip_model.encode_image_tensors(images)
            train_embeddings.append(embeddings)
            train_captions.extend(captions)

    train_embeddings = torch.cat(train_embeddings, dim=0)

    # Pre-compute embeddings for validation data
    print("Pre-computing image embeddings for validation set...")
    val_embeddings = []
    val_captions = []

    with torch.no_grad():
        for images, captions in tqdm(val_loader, desc="Computing val embeddings"):
            if images is None:
                continue
            embeddings = clip_model.encode_image_tensors(images)
            val_embeddings.append(embeddings)
            val_captions.extend(captions)

    val_embeddings = torch.cat(val_embeddings, dim=0)

    print(f"Training with {len(train_captions)} samples")
    print(f"Validating with {len(val_captions)} samples")

    # Train the model
    trained_model, training_losses, bleu_scores = train_caption_model(
        image_embeddings=train_embeddings,
        captions=train_captions,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    return trained_model, training_losses, bleu_scores, val_embeddings, val_captions


def train_caption_model_on_coco(
    clip_model,
    data_dir="data",
    max_samples=1000,
    batch_size=8,
    num_epochs=5,
    split="train2017",
):
    """
    Complete pipeline to train a captioning model on COCO dataset.

    Args:
        clip_model: Pre-trained CLIP model
        data_dir: Directory containing COCO dataset
        max_samples: Maximum number of training samples
        batch_size: Training batch size
        num_epochs: Number of training epochs
        split: COCO split to use ("train2017" or "val2017")

    Returns:
        Tuple of (trained_model, training_losses, bleu_scores)
    """
    # Load COCO dataset
    print("Loading COCO dataset...")
    data_samples = DatasetLoader.load_coco_dataset(
        data_dir=data_dir, split=split, max_samples=max_samples
    )

    print(f"Loaded {len(data_samples)} samples")

    # Split data into train and validation
    train_size = int(0.8 * len(data_samples))
    train_samples = data_samples[:train_size]
    val_samples = data_samples[train_size:]

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    # Train the model
    trained_model, training_losses, bleu_scores = train_caption_model(
        clip_model=clip_model,
        data_samples=train_samples,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    return trained_model, training_losses, bleu_scores, val_samples


def plot_training_metrics(training_losses, bleu_scores):
    """Plot training loss and BLEU scores over epochs."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training loss
    ax1.plot(training_losses, "b-", label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Over Epochs")
    ax1.legend()
    ax1.grid(True)

    # Plot BLEU scores
    ax2.plot(bleu_scores, "r-", label="BLEU Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("BLEU Score")
    ax2.set_title("BLEU Score Over Epochs")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def test_caption_generation(trained_model, clip_model, val_samples, num_samples=5):
    """
    Test caption generation on validation samples.

    Args:
        trained_model: Trained captioning model
        clip_model: CLIP model
        val_samples: Validation samples
        num_samples: Number of samples to test
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Testing caption generation:")
    print("=" * 50)

    for i in range(min(num_samples, len(val_samples))):
        sample = val_samples[i]
        image_tensor = Image.open(sample["image_path"]).convert("RGB").float()
        # Generate image embedding
        image_embedding = (
            clip_model.encode_image_tensors([image_tensor])[0].unsqueeze(0).to(device)
        )

        # Generate caption
        generated_caption = generate_caption(trained_model, image_embedding, tokenizer)

        print(f"Sample {i+1}:")
        print(f"  Ground truth: {sample['text']}")
        print(f"  Generated:    {generated_caption}")
        print("-" * 30)
