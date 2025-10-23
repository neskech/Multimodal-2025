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
    ) -> float:
        """
        1. Finetune a captioning model on the provided embeddings and captions.
        2. Evaluate the model using BLEU score.
        """
        from models.clipCaptionModel import ClipCaptionModel
        from transformers import GPT2Tokenizer

        # Initialize tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        caption_model = ClipCaptionModel(
            prefix_length=10, prefix_size=512, clip_length=512
        )
        caption_model.to(device)

        # Train the captioning model
        trained_model, _, _ = train_caption_model_on_coco(
            clip_model=clip_model, num_epochs=3, batch_size=4
        )

        # Evaluate the trained model
        predictions = []
        for img_emb, _ in embeddings:
            img_emb = img_emb.unsqueeze(0).to(device)
            generated_caption = generate_caption(trained_model, img_emb, tokenizer)
            predictions.append(generated_caption)
        return bleu_score(captions, predictions=predictions)


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
from models.clipCaptionModel import ClipCaptionModel
from typing import List, Tuple, Optional
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CaptionTrainingDataset(Dataset):
    """Dataset for training CLIP captioning model."""

    def __init__(self, data_samples, clip_model, tokenizer, max_length=77):
        """
        Args:
            data_samples: List of data samples from DatasetLoader
            clip_model: CLIP model for generating image embeddings
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.data_samples = data_samples
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-compute image embeddings to avoid recomputing during training
        self.image_embeddings = self._precompute_image_embeddings()

    def _precompute_image_embeddings(self):
        """Pre-compute image embeddings for all samples."""
        print("Pre-computing image embeddings...")
        image_paths = [sample["image_path"] for sample in self.data_samples]

        # Generate embeddings in batches to save memory
        embeddings = []
        batch_size = 32

        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i : i + batch_size]
            batch_embeddings = self.clip_model.encode_images(batch_paths)
            embeddings.extend(batch_embeddings)

        return torch.stack(embeddings)

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        image_embedding = self.image_embeddings[idx]
        caption = sample["text"]

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


def train_caption_model(clip_model, data_samples, num_epochs=5, batch_size=8):
    """
    Train a captioning model on the provided data samples.

    Args:
        clip_model: Pre-trained CLIP model
        data_samples: List of data samples from DatasetLoader
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
    dataset = CaptionTrainingDataset(data_samples, clip_model, tokenizer)
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
            image_embeddings = batch["image_embedding"].to(device)
            tokens = batch["tokens"].to(device)

            optimizer.zero_grad()
            outputs = caption_model(tokens[:, :-1], image_embeddings)
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

        # Evaluate model after each epoch
        bleu = evaluate_captioning_model(
            caption_model, clip_model, data_samples, tokenizer
        )
        bleu_scores.append(bleu)
        print(f"Epoch {epoch+1} BLEU Score: {bleu:.4f}")

    return caption_model, training_losses, bleu_scores


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

        # Generate image embedding
        image_embedding = (
            clip_model.encode_images([sample["image_path"]])[0].unsqueeze(0).to(device)
        )

        # Generate caption
        generated_caption = generate_caption(trained_model, image_embedding, tokenizer)

        print(f"Sample {i+1}:")
        print(f"  Ground truth: {sample['text']}")
        print(f"  Generated:    {generated_caption}")
        print("-" * 30)
