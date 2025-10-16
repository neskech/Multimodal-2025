import json
import os
import torch
import open_clip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import warnings

warnings.filterwarnings("ignore")

# --- Configuration ---
USE_COCO = True
USE_LAION = False
model_name = "ViT-B/32"  # Changed to match the model we used
pretrained_name = "openai"  # Changed to match original CLIP
SAMPLE_SIZE = 10000  # Number of samples to use for GMM fitting
MAX_N_COMPONENTS = 10000  # Maximum number of GMM components to try
MAX_TOTAL_FITS = 20  # Maximum total number of GMM fits to perform
if USE_LAION:
    CACHE_FILE = (
        f"data/clip_embeddings_cache_laion_sample_{model_name.replace('/', '-')}.npz"
    )
elif USE_COCO:
    CACHE_FILE = f"data/clip_embeddings_cache_coco_{model_name.replace('/', '-')}.npz"
else:
    CACHE_FILE = f"data/clip_embeddings_cache_{model_name.replace('/', '-')}.npz"


def load_embeddings():
    """Load embeddings using the same technique as cardelph_projection.py"""
    if os.path.exists(CACHE_FILE):
        cache = np.load(CACHE_FILE, allow_pickle=True)
        embeddings = cache["embeddings"]
        labels = cache["labels"].tolist()
        types = cache["types"].tolist()
        print(f"Loaded cached embeddings from {CACHE_FILE}")
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings, labels, types
    else:
        print(
            f"Cache file {CACHE_FILE} not found. Please run cardelph_projection.py first to generate embeddings."
        )
        return None, None, None


def evaluate_gmm_goodness_of_fit(X, gmm, n_components):
    """Evaluate GMM goodness of fit using multiple metrics"""
    # Predict cluster labels
    labels_pred = gmm.predict(X)

    # Calculate various goodness-of-fit metrics
    results = {
        "n_components": n_components,
        "log_likelihood": gmm.score(X),
        "aic": gmm.aic(X),
        "bic": gmm.bic(X),
        "converged": gmm.converged_,
        "n_iter": gmm.n_iter_,
    }

    # Only calculate clustering metrics if we have more than 1 cluster
    if n_components > 1:
        try:
            results["silhouette_score"] = silhouette_score(X, labels_pred)
            results["calinski_harabasz_score"] = calinski_harabasz_score(X, labels_pred)
            results["davies_bouldin_score"] = davies_bouldin_score(X, labels_pred)
        except Exception as e:
            print(f"Warning: Could not compute clustering metrics: {e}")
            results["silhouette_score"] = np.nan
            results["calinski_harabasz_score"] = np.nan
            results["davies_bouldin_score"] = np.nan
    else:
        results["silhouette_score"] = np.nan
        results["calinski_harabasz_score"] = np.nan
        results["davies_bouldin_score"] = np.nan

    return results


def generate_incremental_components(max_components, max_fits):
    """Generate an incremental sequence of component numbers to test"""
    components = list(
        range(1, max(20, max_components) + 1, max_components // max_fits)
    )  # Dense for small values
    print(f"Initial dense components: {components}")
    return components


def fit_gmm_with_model_selection(X, max_components=10):
    """Fit GMM with different numbers of components using incremental approach"""
    print(
        f"Fitting GMM with up to {max_components} components using incremental approach..."
    )

    # Generate incremental component sequence
    component_sequence = generate_incremental_components(max_components, MAX_TOTAL_FITS)

    print(
        f"Testing {len(component_sequence)} different component counts: {component_sequence[:10]}{'...' if len(component_sequence) > 10 else ''}"
    )
    print(f"Full sequence: {component_sequence}")

    results = []
    models = {}

    for i, n_components in enumerate(component_sequence):
        print(
            f"Fitting GMM {i+1}/{len(component_sequence)}: {n_components} components..."
        )

        # Fit GMM
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=42,
            max_iter=100,
            tol=1e-3,
        )

        try:
            gmm.fit(X)
            result = evaluate_gmm_goodness_of_fit(X, gmm, n_components)
            results.append(result)
            models[n_components] = gmm

            print(f"  Components: {n_components}")
            print(f"  Log-likelihood: {result['log_likelihood']:.2f}")
            print(f"  AIC: {result['aic']:.2f}")
            print(f"  BIC: {result['bic']:.2f}")
            print(f"  Converged: {result['converged']}")

        except Exception as e:
            print(f"  Failed to fit GMM with {n_components} components: {e}")
            continue

    return results, models


def visualize_gmm_results(X, gmm, labels_true=None, title="GMM Clustering Results"):
    """Visualize GMM results in 2D using PCA"""
    # Reduce to 2D for visualization
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance ratio: {explained_var:.3f}")
    else:
        X_2d = X
        explained_var = 1.0

    # Predict cluster labels and probabilities
    labels_pred = gmm.predict(X)
    proba = gmm.predict_proba(X)

    # Create subplot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Cluster assignments
    scatter = axes[0].scatter(
        X_2d[:, 0], X_2d[:, 1], c=labels_pred, cmap="viridis", alpha=0.6
    )
    axes[0].set_title(f"{title}\nCluster Assignments ({gmm.n_components} components)")
    axes[0].set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)"
        if X.shape[1] > 2
        else "Dim 1"
    )
    axes[0].set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)"
        if X.shape[1] > 2
        else "Dim 2"
    )
    plt.colorbar(scatter, ax=axes[0])

    # Plot 2: Uncertainty (entropy of probabilities)
    entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
    scatter2 = axes[1].scatter(
        X_2d[:, 0], X_2d[:, 1], c=entropy, cmap="plasma", alpha=0.6
    )
    axes[1].set_title("Assignment Uncertainty\n(Higher = More Uncertain)")
    axes[1].set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)"
        if X.shape[1] > 2
        else "Dim 1"
    )
    axes[1].set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)"
        if X.shape[1] > 2
        else "Dim 2"
    )
    plt.colorbar(scatter2, ax=axes[1])

    plt.tight_layout()
    plt.show()


def print_model_selection_summary(results):
    """Print a summary table of model selection results"""
    print("\n" + "=" * 80)
    print("GMM MODEL SELECTION SUMMARY")
    print("=" * 80)
    print(
        f"{'Components':<12} {'Log-Lik':<12} {'AIC':<12} {'BIC':<12} {'Silhouette':<12} {'Converged':<12}"
    )
    print("-" * 80)

    best_bic_idx = np.argmin([r["bic"] for r in results])
    best_aic_idx = np.argmin([r["aic"] for r in results])
    best_loglik_idx = np.argmax([r["log_likelihood"] for r in results])

    # Show first few, then best models if not already shown, then last few
    total_results = len(results)
    show_first = min(5, total_results)
    show_last = min(3, total_results)

    indices_to_show = list(range(show_first))

    # Add best model indices if not already included
    best_indices = [best_bic_idx, best_aic_idx, best_loglik_idx]
    for idx in best_indices:
        if idx not in indices_to_show:
            indices_to_show.append(idx)

    # Add last few if not already included
    if total_results > show_first:
        for idx in range(max(show_first, total_results - show_last), total_results):
            if idx not in indices_to_show:
                indices_to_show.append(idx)

    indices_to_show = sorted(list(set(indices_to_show)))

    prev_idx = -1
    for i in indices_to_show:
        if i > prev_idx + 1 and prev_idx >= 0:
            print("    ...")

        result = results[i]
        components = result["n_components"]
        log_lik = result["log_likelihood"]
        aic = result["aic"]
        bic = result["bic"]
        silhouette = result["silhouette_score"]
        converged = "Yes" if result["converged"] else "No"

        # Mark best models
        markers = []
        if i == best_bic_idx:
            markers.append("BIC*")
        if i == best_aic_idx:
            markers.append("AIC*")
        if i == best_loglik_idx:
            markers.append("LL*")

        marker_str = " ".join(markers)

        silhouette_str = f"{silhouette:.3f}" if not np.isnan(silhouette) else "N/A"

        print(
            f"{components:<12} {log_lik:<12.2f} {aic:<12.1f} {bic:<12.1f} {silhouette_str:<12} {converged:<12} {marker_str}"
        )
        prev_idx = i

    print("-" * 80)
    print(
        f"Tested {total_results} different component counts using incremental approach"
    )
    print("* Best model according to: BIC (Bayesian Information Criterion),")
    print("  AIC (Akaike Information Criterion), LL (Log-Likelihood)")
    print("\nRecommendation: BIC is generally preferred for model selection.")


def analyze_embeddings_by_type(
    embeddings, labels, types, embedding_type, max_components=MAX_N_COMPONENTS
):
    """Analyze embeddings of a specific type (image or text)"""
    print(f"\n{'='*60}")
    print(f"GMM FITTING FOR {embedding_type.upper()} EMBEDDINGS ONLY")
    print(f"{'='*60}")

    # Filter embeddings by type
    type_mask = np.array([t == embedding_type for t in types])
    X = embeddings[type_mask]
    filtered_labels = [labels[i] for i in range(len(labels)) if type_mask[i]]

    if len(X) == 0:
        print(f"No {embedding_type} embeddings found!")
        return None, None

    print(f"Total {embedding_type} embeddings: {len(X)}")
    print(f"Embedding dimension: {X.shape[1]}")

    # Sample if needed
    if len(X) > SAMPLE_SIZE:
        print(
            f"Sampling {SAMPLE_SIZE} {embedding_type} embeddings from {len(X)} total..."
        )
        indices = np.random.choice(len(X), SAMPLE_SIZE, replace=False)
        X = X[indices]
        filtered_labels = [filtered_labels[i] for i in indices]

    print(f"Using {len(X)} {embedding_type} samples for GMM fitting")

    # Normalize embeddings
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / norms

    print(f"{embedding_type.title()} embeddings normalized to unit vectors")

    # Fit GMM with model selection
    results, models = fit_gmm_with_model_selection(
        X_normalized, max_components=MAX_N_COMPONENTS
    )

    if not results:
        print(
            f"No GMM models were successfully fitted for {embedding_type} embeddings."
        )
        return None, None

    # Print model selection summary
    print(f"\n{embedding_type.upper()} EMBEDDINGS - MODEL SELECTION SUMMARY")
    print_model_selection_summary(results)

    # Select best model based on AIC
    best_model_idx = np.argmin([r["aic"] for r in results])
    best_n_components = results[best_model_idx]["n_components"]
    best_gmm = models[best_n_components]

    print(f"\nBest {embedding_type} model: {best_n_components} components (lowest AIC)")
    print(f"BIC: {results[best_model_idx]['bic']:.1f}")
    print(f"Log-likelihood: {results[best_model_idx]['log_likelihood']:.2f}")

    if not np.isnan(results[best_model_idx]["silhouette_score"]):
        print(f"Silhouette score: {results[best_model_idx]['silhouette_score']:.3f}")

    # Visualize the best model
    visualize_gmm_results(
        X_normalized,
        best_gmm,
        filtered_labels,
        f"Best {embedding_type.title()} GMM Model ({best_n_components} components)",
    )

    return results, X_normalized


def compare_image_text_statistics(embeddings, types):
    """Compare basic statistics between image and text embeddings"""
    print(f"\n{'='*60}")
    print("COMPARING IMAGE vs TEXT EMBEDDING STATISTICS")
    print(f"{'='*60}")

    # Separate embeddings by type
    image_mask = np.array([t == "image" for t in types])
    text_mask = np.array([t == "text" for t in types])

    image_embeddings = embeddings[image_mask]
    text_embeddings = embeddings[text_mask]

    if len(image_embeddings) == 0 or len(text_embeddings) == 0:
        print("Missing image or text embeddings for comparison!")
        return

    # Normalize embeddings
    image_norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    image_normalized = image_embeddings / image_norms
    text_normalized = text_embeddings / text_norms

    print(f"Image embeddings: {len(image_embeddings)} samples")
    print(f"Text embeddings: {len(text_embeddings)} samples")
    print()

    # Basic statistics
    print("BASIC STATISTICS:")
    print("-" * 40)
    print(f"{'Metric':<25} {'Image':<15} {'Text':<15}")
    print("-" * 40)
    print(
        f"{'Mean norm (original)':<25} {np.mean(image_norms.flatten()):<15.4f} {np.mean(text_norms.flatten()):<15.4f}"
    )
    print(
        f"{'Std norm (original)':<25} {np.std(image_norms.flatten()):<15.4f} {np.std(text_norms.flatten()):<15.4f}"
    )
    print(
        f"{'Mean (normalized)':<25} {np.mean(image_normalized):<15.4f} {np.mean(text_normalized):<15.4f}"
    )
    print(
        f"{'Std (normalized)':<25} {np.std(image_normalized):<15.4f} {np.std(text_normalized):<15.4f}"
    )

    # Compute pairwise distances within each type
    from scipy.spatial.distance import pdist

    # Sample for computational efficiency if needed
    sample_size = min(1000, len(image_embeddings), len(text_embeddings))

    image_sample = image_normalized[
        np.random.choice(len(image_normalized), sample_size, replace=False)
    ]
    text_sample = text_normalized[
        np.random.choice(len(text_normalized), sample_size, replace=False)
    ]

    image_distances = pdist(image_sample, metric="cosine")
    text_distances = pdist(text_sample, metric="cosine")

    print(
        f"{'Mean cosine distance':<25} {np.mean(image_distances):<15.4f} {np.mean(text_distances):<15.4f}"
    )
    print(
        f"{'Std cosine distance':<25} {np.std(image_distances):<15.4f} {np.std(text_distances):<15.4f}"
    )


def main():
    print("=" * 60)
    print("GMM FITTING FOR CLIP EMBEDDINGS")
    print("=" * 60)
    print(f"Using incremental approach to test up to {MAX_N_COMPONENTS} components")
    print(f"Maximum total fits: {MAX_TOTAL_FITS}")
    print(f"Component sequence: dense for small values, sparse for large values")
    print("=" * 60)

    # Load embeddings
    embeddings, labels, types = load_embeddings()

    if embeddings is None:
        print("Failed to load embeddings. Exiting.")
        return

    print(f"Total embeddings loaded: {len(embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Types available: {set(types)}")

    # Compare basic statistics between image and text embeddings
    compare_image_text_statistics(embeddings, types)

    # Analyze image embeddings separately
    image_results, image_embeddings_norm = analyze_embeddings_by_type(
        embeddings, labels, types, "image", max_components=MAX_N_COMPONENTS
    )

    # Analyze text embeddings separately
    text_results, text_embeddings_norm = analyze_embeddings_by_type(
        embeddings, labels, types, "text", max_components=MAX_N_COMPONENTS
    )

    # Combined analysis for comparison
    print(f"\n{'='*60}")
    print("COMBINED ANALYSIS (ORIGINAL)")
    print(f"{'='*60}")

    # Sample embeddings for efficiency if dataset is large
    if len(embeddings) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} embeddings from {len(embeddings)} total...")
        indices = np.random.choice(len(embeddings), SAMPLE_SIZE, replace=False)
        X = embeddings[indices]
        sampled_labels = [labels[i] for i in indices]
        sampled_types = [types[i] for i in indices]
    else:
        X = embeddings
        sampled_labels = labels
        sampled_types = types

    print(f"Using {len(X)} samples for combined GMM fitting")

    # Normalize embeddings (common practice for CLIP embeddings)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / norms

    print(f"Combined embeddings normalized to unit vectors")

    # Fit GMM with model selection
    results, models = fit_gmm_with_model_selection(
        X_normalized, max_components=MAX_N_COMPONENTS
    )

    if not results:
        print("No GMM models were successfully fitted.")
        return

    # Print model selection summary
    print("\nCOMBINED EMBEDDINGS - MODEL SELECTION SUMMARY")
    print_model_selection_summary(results)

    # Select best model based on BIC
    best_model_idx = np.argmin([r["bic"] for r in results])
    best_n_components = results[best_model_idx]["n_components"]
    best_gmm = models[best_n_components]

    print(f"\nBest combined model: {best_n_components} components (lowest BIC)")
    print(f"BIC: {results[best_model_idx]['bic']:.1f}")
    print(f"Log-likelihood: {results[best_model_idx]['log_likelihood']:.2f}")

    if not np.isnan(results[best_model_idx]["silhouette_score"]):
        print(f"Silhouette score: {results[best_model_idx]['silhouette_score']:.3f}")

    # Visualize the best model
    visualize_gmm_results(
        X_normalized,
        best_gmm,
        sampled_labels,
        f"Best Combined GMM Model ({best_n_components} components)",
    )

    # Additional analysis: cluster composition for combined model
    print("\n" + "=" * 60)
    print("COMBINED MODEL - CLUSTER COMPOSITION ANALYSIS")
    print("=" * 60)

    cluster_labels = best_gmm.predict(X_normalized)
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_types = [sampled_types[i] for i in range(len(sampled_types)) if mask[i]]
        cluster_size = np.sum(mask)

        type_counts = {}
        for t in cluster_types:
            type_counts[t] = type_counts.get(t, 0) + 1

        print(f"\nCluster {cluster_id}: {cluster_size} samples")
        for t, count in type_counts.items():
            percentage = (count / cluster_size) * 100
            print(f"  {t}: {count} ({percentage:.1f}%)")

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")

    if image_results and text_results:
        image_best_idx = np.argmin([r["bic"] for r in image_results])
        text_best_idx = np.argmin([r["bic"] for r in text_results])
        combined_best_idx = np.argmin([r["bic"] for r in results])

        print(f"{'Model':<20} {'Components':<12} {'BIC':<15} {'Log-Likelihood':<15}")
        print("-" * 62)
        print(
            f"{'Image Only':<20} {image_results[image_best_idx]['n_components']:<12} {image_results[image_best_idx]['bic']:<15.1f} {image_results[image_best_idx]['log_likelihood']:<15.2f}"
        )
        print(
            f"{'Text Only':<20} {text_results[text_best_idx]['n_components']:<12} {text_results[text_best_idx]['bic']:<15.1f} {text_results[text_best_idx]['log_likelihood']:<15.2f}"
        )
        print(
            f"{'Combined':<20} {results[combined_best_idx]['n_components']:<12} {results[combined_best_idx]['bic']:<15.1f} {results[combined_best_idx]['log_likelihood']:<15.2f}"
        )

    print("\n" + "=" * 60)
    print("GMM FITTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
