#!/usr/bin/env python3
"""
Example usage of generate_embeddings.py for different datasets.
"""

from generate_embeddings import EmbeddingGenerator, DatasetLoader, save_embeddings
import numpy as np
import os


def example_coco_embeddings():
    """Generate embeddings for COCO dataset."""
    print("=" * 60)
    print("Generating COCO Dataset Embeddings")
    print("=" * 60)

    # Initialize generator
    generator = EmbeddingGenerator(model_name="ViT-B/32", use_openclip=False)

    # Load COCO data (small sample)
    try:
        data = DatasetLoader.load_coco_dataset(
            data_dir="data", split="train2017", max_samples=50  # Small sample for demo
        )

        # Extract texts and image paths
        texts = [item["text"] for item in data]
        image_paths = [item["image_path"] for item in data]
        labels = [str(item["image_id"]) for item in data]

        # Generate embeddings
        print("Generating text embeddings...")
        text_embeddings = generator.generate_text_embeddings(texts)

        print("Generating image embeddings...")
        image_embeddings = generator.generate_image_embeddings(image_paths)

        # Combine embeddings
        all_embeddings = np.vstack([image_embeddings, text_embeddings])
        all_labels = labels + labels  # Same labels for image-text pairs
        all_types = ["image"] * len(image_embeddings) + ["text"] * len(text_embeddings)

        # Save embeddings
        cache_file = "data/example_coco_embeddings.npz"
        metadata = {
            "dataset": "coco",
            "model_name": "ViT-B/32",
            "n_samples": len(all_embeddings),
        }

        save_embeddings(all_embeddings, all_labels, all_types, metadata, cache_file)
        print(f"Saved {len(all_embeddings)} embeddings to {cache_file}")

    except FileNotFoundError as e:
        print(f"COCO dataset not found: {e}")
        print("Download COCO dataset first or use a different example.")


def example_laion_sample_embeddings():
    """Generate embeddings for LAION sample (text-only)."""
    print("=" * 60)
    print("Generating LAION Sample Embeddings")
    print("=" * 60)

    # Initialize generator
    generator = EmbeddingGenerator(model_name="ViT-B/32", use_openclip=False)

    # Load LAION sample data
    data = DatasetLoader.load_laion_sample()

    # Extract texts
    texts = [item["text"] for item in data]
    labels = [str(item["sample_id"]) for item in data]

    # Generate embeddings
    print("Generating text embeddings...")
    text_embeddings = generator.generate_text_embeddings(texts)

    # Save embeddings
    cache_file = "data/example_laion_embeddings.npz"
    metadata = {
        "dataset": "laion_sample",
        "model_name": "ViT-B/32",
        "n_samples": len(text_embeddings),
    }

    save_embeddings(
        text_embeddings, labels, ["text"] * len(text_embeddings), metadata, cache_file
    )
    print(f"Saved {len(text_embeddings)} embeddings to {cache_file}")


def example_custom_text_embeddings():
    """Generate embeddings for custom text data."""
    print("=" * 60)
    print("Generating Custom Text Embeddings")
    print("=" * 60)

    # Create a sample text file
    sample_texts = [
        "A beautiful sunset over the ocean",
        "A cat playing with a ball of yarn",
        "A modern city skyline at night",
        "A forest with tall pine trees",
        "A delicious pizza with various toppings",
    ]

    text_file = "data/sample_texts.txt"
    os.makedirs("data", exist_ok=True)

    with open(text_file, "w") as f:
        for text in sample_texts:
            f.write(text + "\\n")

    print(f"Created sample text file: {text_file}")

    # Initialize generator
    generator = EmbeddingGenerator(model_name="ViT-B/32", use_openclip=False)

    # Load custom text data
    data = DatasetLoader.load_text_file(text_file)

    # Extract texts
    texts = [item["text"] for item in data]
    labels = [str(item["text_id"]) for item in data]

    # Generate embeddings
    print("Generating text embeddings...")
    text_embeddings = generator.generate_text_embeddings(texts)

    # Save embeddings
    cache_file = "data/example_custom_text_embeddings.npz"
    metadata = {
        "dataset": "custom_text",
        "model_name": "ViT-B/32",
        "n_samples": len(text_embeddings),
    }

    save_embeddings(
        text_embeddings, labels, ["text"] * len(text_embeddings), metadata, cache_file
    )
    print(f"Saved {len(text_embeddings)} embeddings to {cache_file}")


def main():
    """Run examples for different datasets."""
    print("CLIP Embedding Generation Examples")
    print("Choose which example to run:")
    print("1. COCO Dataset (requires COCO data)")
    print("2. LAION Sample (text-only)")
    print("3. Custom Text")
    print("4. All examples")

    choice = input("Enter your choice (1-4): ").strip()

    if choice == "1":
        example_coco_embeddings()
    elif choice == "2":
        example_laion_sample_embeddings()
    elif choice == "3":
        example_custom_text_embeddings()
    elif choice == "4":
        example_laion_sample_embeddings()
        example_custom_text_embeddings()
        example_coco_embeddings()
    else:
        print("Invalid choice. Running LAION sample example by default.")
        example_laion_sample_embeddings()


if __name__ == "__main__":
    main()
