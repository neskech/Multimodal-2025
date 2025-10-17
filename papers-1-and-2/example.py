"""
Example script demonstrating von Mises-Fisher mixture modeling on CLIP embeddings.
"""

import numpy as np
from clip_embeddings import CLIPEmbeddingExtractor, create_sample_data
from em_algorithm import fit_von_mises_fisher_mixture
from visualization import visualize_mixture_model


def main():
    """Run a simple example with sample CLIP embeddings."""
    print("Von Mises-Fisher Mixture Model Example")
    print("=" * 50)

    # Initialize CLIP extractor
    print("Initializing CLIP model...")
    extractor = CLIPEmbeddingExtractor(model_name="ViT-B/32")

    # Generate sample CLIP embeddings
    print("Generating sample CLIP embeddings...")
    embeddings = create_sample_data(extractor, num_samples=500)
    print(
        f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}"
    )

    # Fit mixture model
    print("\nFitting von Mises-Fisher mixture model...")
    mixture_model, convergence_info = fit_von_mises_fisher_mixture(
        embeddings, n_components=3, max_iterations=50, verbose=True
    )

    # Print results
    print(f"\nConvergence: {convergence_info['converged']}")
    print(f"Iterations: {convergence_info['n_iterations']}")
    print(f"Final log-likelihood: {convergence_info['final_log_likelihood']:.4f}")

    # Get fitted parameters
    weights, mean_directions, concentrations = mixture_model.get_parameters()
    print(f"\nFitted Parameters:")
    for i, (w, c) in enumerate(zip(weights, concentrations)):
        print(f"  Component {i}: weight={w:.4f}, concentration={c:.4f}")

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_mixture_model(mixture_model, embeddings, "example_visualizations")

    print("\nExample completed! Check 'example_visualizations' directory for results.")


if __name__ == "__main__":
    main()
