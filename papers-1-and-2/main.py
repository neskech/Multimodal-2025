"""
Main script for fitting von Mises-Fisher mixture models to CLIP embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import List, Optional, Tuple
import json
from datetime import datetime

from clip_embeddings import CLIPEmbeddingExtractor, create_sample_data
from von_mises_fisher import VonMisesFisherMixture
from em_algorithm import fit_von_mises_fisher_mixture
from visualization import visualize_mixture_model


def load_text_data(file_path: str) -> List[str]:
    """Load text data from a file (one text per line)."""
    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts


def load_image_paths(directory: str) -> List[str]:
    """Load image file paths from a directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths


def evaluate_mixture_model(
    mixture_model: VonMisesFisherMixture, test_data: np.ndarray
) -> dict:
    """
    Evaluate the fitted mixture model.

    Args:
        mixture_model: Fitted von Mises-Fisher mixture model
        test_data: Test data points

    Returns:
        Dictionary with evaluation metrics
    """
    # Compute log-likelihood on test data
    log_likelihood = np.sum(mixture_model.log_pdf(test_data))

    # Compute average log-likelihood per sample
    avg_log_likelihood = log_likelihood / len(test_data)

    # Compute perplexity (exponential of negative average log-likelihood)
    perplexity = np.exp(-avg_log_likelihood)

    # Get model parameters
    weights, mean_directions, concentrations = mixture_model.get_parameters()

    # Compute component statistics
    component_stats = []
    for i, (weight, mean_dir, conc) in enumerate(
        zip(weights, mean_directions, concentrations)
    ):
        component_stats.append(
            {
                "component_id": i,
                "weight": float(weight),
                "concentration": float(conc),
                "mean_direction": mean_dir.tolist(),
            }
        )

    return {
        "log_likelihood": float(log_likelihood),
        "avg_log_likelihood": float(avg_log_likelihood),
        "perplexity": float(perplexity),
        "n_components": mixture_model.n_components,
        "component_stats": component_stats,
    }


def save_results(
    mixture_model: VonMisesFisherMixture,
    convergence_info: dict,
    evaluation_metrics: dict,
    output_dir: str,
):
    """Save fitted model and results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save model parameters
    weights, mean_directions, concentrations = mixture_model.get_parameters()

    model_params = {
        "weights": weights.tolist(),
        "mean_directions": mean_directions.tolist(),
        "concentrations": concentrations.tolist(),
        "n_components": mixture_model.n_components,
        "dimension": mixture_model.dimension,
    }

    with open(os.path.join(output_dir, "model_parameters.json"), "w") as f:
        json.dump(model_params, f, indent=2)

    # Save convergence info
    with open(os.path.join(output_dir, "convergence_info.json"), "w") as f:
        json.dump(convergence_info, f, indent=2)

    # Save evaluation metrics
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(evaluation_metrics, f, indent=2)

    # Save log-likelihood history plot
    if convergence_info["log_likelihood_history"]:
        plt.figure(figsize=(10, 6))
        plt.plot(convergence_info["log_likelihood_history"])
        plt.xlabel("Iteration")
        plt.ylabel("Log-Likelihood")
        plt.title("EM Algorithm Convergence")
        plt.grid(True)
        plt.savefig(
            os.path.join(output_dir, "convergence_plot.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit von Mises-Fisher mixture model to CLIP embeddings"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=10,
        help="Number of mixture components (default: 5)",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="Maximum EM iterations (default: 100)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Convergence tolerance (default: 1e-6)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="Path to text file with one text per line",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to directory containing images",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="Number of sample embeddings to generate if no data provided (default: 1000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        help="CLIP model to use (default: ViT-B/32)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Von Mises-Fisher Mixture Model for CLIP Embeddings")
    print("=" * 60)

    # Initialize CLIP extractor
    print(f"Initializing CLIP model: {args.clip_model}")
    extractor = CLIPEmbeddingExtractor(model_name=args.clip_model)
    print(f"CLIP embedding dimension: {extractor.get_embedding_dimension()}")

    # Load or generate data
    embeddings = None

    if args.text_file and os.path.exists(args.text_file):
        print(f"Loading text data from: {args.text_file}")
        texts = load_text_data(args.text_file)
        print(f"Loaded {len(texts)} text samples")
        embeddings = extractor.extract_text_embeddings(texts)

    elif args.image_dir and os.path.exists(args.image_dir):
        print(f"Loading image data from: {args.image_dir}")
        image_paths = load_image_paths(args.image_dir)
        print(f"Found {len(image_paths)} images")
        if len(image_paths) > 0:
            embeddings = extractor.extract_image_embeddings(image_paths)
        else:
            print("No images found, generating sample data...")
            embeddings = create_sample_data(extractor, args.n_samples)
    else:
        print("No data provided, generating sample CLIP embeddings...")
        embeddings = create_sample_data(extractor, args.n_samples)

    print(f"Data shape: {embeddings.shape}")
    print(
        f"Embedding norms (should be ~1.0): mean={np.mean(np.linalg.norm(embeddings, axis=1)):.4f}, std={np.std(np.linalg.norm(embeddings, axis=1)):.4f}"
    )

    # Split data into train/test
    n_samples = len(embeddings)
    n_test = int(n_samples * args.test_split)
    n_train = n_samples - n_test

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_data = embeddings[train_indices]
    test_data = embeddings[test_indices]

    print(f"Training data: {train_data.shape}")
    print(f"Test data: {test_data.shape}")

    # Fit mixture model
    print(
        f"\nFitting von Mises-Fisher mixture model with {args.n_components} components..."
    )
    print(
        f"EM parameters: max_iterations={args.max_iterations}, tolerance={args.tolerance}"
    )

    mixture_model, convergence_info = fit_von_mises_fisher_mixture(
        train_data,
        n_components=args.n_components,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        random_state=args.random_state,
        verbose=True,
    )

    # Evaluate model
    print("\nEvaluating fitted model...")
    evaluation_metrics = evaluate_mixture_model(mixture_model, test_data)

    print(f"\nEvaluation Results:")
    print(f"Test Log-Likelihood: {evaluation_metrics['log_likelihood']:.4f}")
    print(f"Average Log-Likelihood: {evaluation_metrics['avg_log_likelihood']:.4f}")
    print(f"Perplexity: {evaluation_metrics['perplexity']:.4f}")

    print(f"\nComponent Statistics:")
    for stats in evaluation_metrics["component_stats"]:
        print(
            f"  Component {stats['component_id']}: weight={stats['weight']:.4f}, concentration={stats['concentration']:.4f}"
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    save_results(mixture_model, convergence_info, evaluation_metrics, output_dir)

    # Create visualizations
    print("\nCreating visualizations...")
    viz_dir = os.path.join(output_dir, "visualizations")
    visualize_mixture_model(mixture_model, test_data, viz_dir)

    print(f"\nFitting completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    main()
