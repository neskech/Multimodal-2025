"""
Visualization tools for von Mises-Fisher mixture models on CLIP embeddings.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, Tuple, List
import warnings

from von_mises_fisher import VonMisesFisherMixture


class VonMisesFisherVisualizer:
    """Visualization tools for von Mises-Fisher mixture models."""

    def __init__(self, mixture_model: VonMisesFisherMixture):
        """
        Initialize visualizer.

        Args:
            mixture_model: Fitted von Mises-Fisher mixture model
        """
        self.mixture_model = mixture_model
        self.weights, self.mean_directions, self.concentrations = (
            mixture_model.get_parameters()
        )

    def plot_component_parameters(self, figsize: Tuple[int, int] = (15, 5)):
        """
        Plot component parameters (weights and concentrations).

        Args:
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Plot weights
        axes[0].bar(range(len(self.weights)), self.weights)
        axes[0].set_xlabel("Component")
        axes[0].set_ylabel("Weight")
        axes[0].set_title("Component Weights")
        axes[0].grid(True, alpha=0.3)

        # Plot concentrations
        axes[1].bar(range(len(self.concentrations)), self.concentrations)
        axes[1].set_xlabel("Component")
        axes[1].set_ylabel("Concentration")
        axes[1].set_title("Component Concentrations")
        axes[1].grid(True, alpha=0.3)

        # Plot weight vs concentration
        axes[2].scatter(self.weights, self.concentrations, s=100, alpha=0.7)
        axes[2].set_xlabel("Weight")
        axes[2].set_ylabel("Concentration")
        axes[2].set_title("Weight vs Concentration")
        axes[2].grid(True, alpha=0.3)

        # Add component labels
        for i, (w, c) in enumerate(zip(self.weights, self.concentrations)):
            axes[2].annotate(f"C{i}", (w, c), xytext=(5, 5), textcoords="offset points")

        plt.tight_layout()
        return fig

    def plot_mean_directions_2d(
        self,
        data: Optional[np.ndarray] = None,
        method: str = "pca",
        figsize: Tuple[int, int] = (10, 8),
    ):
        """
        Plot mean directions in 2D using dimensionality reduction.

        Args:
            data: Data points to plot alongside mean directions
            method: Dimensionality reduction method ('pca' or 'tsne')
            figsize: Figure size
        """
        if method == "pca":
            reducer = PCA(n_components=2)
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")

        # Prepare data for reduction
        all_points = [self.mean_directions]
        labels = ["mean_directions"]

        if data is not None:
            all_points.append(data)
            labels.append("data")

        combined_data = np.vstack(all_points)

        # Apply dimensionality reduction
        reduced_data = reducer.fit_transform(combined_data)

        # Split back
        n_means = len(self.mean_directions)
        mean_dirs_2d = reduced_data[:n_means]
        data_2d = reduced_data[n_means:] if data is not None else None

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot data points
        if data_2d is not None:
            ax.scatter(
                data_2d[:, 0],
                data_2d[:, 1],
                alpha=0.3,
                s=20,
                c="lightblue",
                label="Data points",
            )

        # Plot mean directions
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.mean_directions)))
        for i, (mean_dir_2d, weight, conc) in enumerate(
            zip(mean_dirs_2d, self.weights, self.concentrations)
        ):
            # Size proportional to weight, color intensity to concentration
            size = weight * 1000
            alpha = min(1.0, conc / 10.0)  # Normalize concentration for alpha

            ax.scatter(
                mean_dir_2d[0],
                mean_dir_2d[1],
                s=size,
                c=[colors[i]],
                alpha=alpha,
                edgecolors="black",
                linewidth=2,
                label=f"Component {i} (w={weight:.3f}, k={conc:.2f})",
            )

        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_title(f"Mean Directions in 2D ({method.upper()})")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_mean_directions_3d(
        self, data: Optional[np.ndarray] = None, figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Plot mean directions in 3D (first 3 dimensions).

        Args:
            data: Data points to plot alongside mean directions
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Plot data points
        if data is not None:
            ax.scatter(
                data[:, 0],
                data[:, 1],
                data[:, 2],
                alpha=0.2,
                s=10,
                c="lightblue",
                label="Data points",
            )

        # Plot mean directions
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.mean_directions)))
        for i, (mean_dir, weight, conc) in enumerate(
            zip(self.mean_directions, self.weights, self.concentrations)
        ):
            # Size proportional to weight
            size = weight * 1000

            ax.scatter(
                mean_dir[0],
                mean_dir[1],
                mean_dir[2],
                s=size,
                c=[colors[i]],
                alpha=0.8,
                edgecolors="black",
                linewidth=2,
                label=f"Component {i} (w={weight:.3f}, k={conc:.2f})",
            )

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Dimension 3")
        ax.set_title("Mean Directions in 3D (First 3 Dimensions)")
        ax.legend()

        return fig

    def plot_component_similarity_matrix(self, figsize: Tuple[int, int] = (8, 6)):
        """
        Plot similarity matrix between component mean directions.

        Args:
            figsize: Figure size
        """
        # Compute cosine similarities between mean directions
        similarity_matrix = np.dot(self.mean_directions, self.mean_directions.T)

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap="coolwarm", vmin=-1, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Cosine Similarity")

        # Add text annotations
        for i in range(len(self.mean_directions)):
            for j in range(len(self.mean_directions)):
                text = ax.text(
                    j,
                    i,
                    f"{similarity_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        ax.set_xlabel("Component")
        ax.set_ylabel("Component")
        ax.set_title("Component Mean Direction Similarities")
        ax.set_xticks(range(len(self.mean_directions)))
        ax.set_yticks(range(len(self.mean_directions)))

        plt.tight_layout()
        return fig

    def plot_data_assignment(
        self, data: np.ndarray, method: str = "pca", figsize: Tuple[int, int] = (12, 5)
    ):
        """
        Plot data points colored by their most likely component assignment.

        Args:
            data: Data points
            method: Dimensionality reduction method
            figsize: Figure size
        """
        # Compute component assignments
        n_samples = data.shape[0]
        responsibilities = np.zeros((n_samples, self.mixture_model.n_components))

        for k in range(self.mixture_model.n_components):
            component = self.mixture_model.components[k]
            log_probs = component.log_pdf(data)
            responsibilities[:, k] = np.log(self.weights[k]) + log_probs

        # Get most likely assignments
        assignments = np.argmax(responsibilities, axis=1)

        # Apply dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2)
        elif method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")

        data_2d = reducer.fit_transform(data)

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Data colored by assignment
        colors = plt.cm.Set3(np.linspace(0, 1, self.mixture_model.n_components))
        for k in range(self.mixture_model.n_components):
            mask = assignments == k
            if np.any(mask):
                ax1.scatter(
                    data_2d[mask, 0],
                    data_2d[mask, 1],
                    c=[colors[k]],
                    alpha=0.6,
                    s=20,
                    label=f"Component {k}",
                )

        ax1.set_xlabel(f"{method.upper()} Component 1")
        ax1.set_ylabel(f"{method.upper()} Component 2")
        ax1.set_title("Data Points by Component Assignment")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Assignment distribution
        assignment_counts = np.bincount(
            assignments, minlength=self.mixture_model.n_components
        )
        ax2.bar(range(len(assignment_counts)), assignment_counts)
        ax2.set_xlabel("Component")
        ax2.set_ylabel("Number of Assigned Points")
        ax2.set_title("Component Assignment Distribution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_component_entropy(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot entropy of each component.

        Args:
            figsize: Figure size
        """
        entropies = []
        for component in self.mixture_model.components:
            entropies.append(component.entropy())

        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.bar(range(len(entropies)), entropies)
        ax.set_xlabel("Component")
        ax.set_ylabel("Entropy")
        ax.set_title("Component Entropies")
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, entropy) in enumerate(zip(bars, entropies)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{entropy:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        return fig

    def create_comprehensive_visualization(
        self, data: Optional[np.ndarray] = None, output_path: Optional[str] = None
    ):
        """
        Create a comprehensive visualization of the mixture model.

        Args:
            data: Data points to include in visualization
            output_path: Path to save the visualization
        """
        # Create subplots
        fig = plt.figure(figsize=(20, 15))

        # 1. Component parameters
        ax1 = plt.subplot(3, 4, 1)
        ax1.bar(range(len(self.weights)), self.weights)
        ax1.set_title("Component Weights")
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Weight")
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(3, 4, 2)
        ax2.bar(range(len(self.concentrations)), self.concentrations)
        ax2.set_title("Component Concentrations")
        ax2.set_xlabel("Component")
        ax2.set_ylabel("Concentration")
        ax2.grid(True, alpha=0.3)

        # 2. Mean directions in 2D (PCA)
        ax3 = plt.subplot(3, 4, 3)
        if data is not None:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(data)
            ax3.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.3, s=10, c="lightblue")

        mean_dirs_2d = pca.transform(self.mean_directions)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.mean_directions)))
        for i, (mean_dir_2d, weight) in enumerate(zip(mean_dirs_2d, self.weights)):
            size = weight * 1000
            ax3.scatter(
                mean_dir_2d[0],
                mean_dir_2d[1],
                s=size,
                c=[colors[i]],
                alpha=0.8,
                edgecolors="black",
                linewidth=2,
            )
        ax3.set_title("Mean Directions (PCA)")
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.grid(True, alpha=0.3)

        # 3. Similarity matrix
        ax4 = plt.subplot(3, 4, 4)
        similarity_matrix = np.dot(self.mean_directions, self.mean_directions.T)
        im = ax4.imshow(similarity_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax4.set_title("Component Similarities")
        ax4.set_xlabel("Component")
        ax4.set_ylabel("Component")
        plt.colorbar(im, ax=ax4)

        # 4. Component entropies
        ax5 = plt.subplot(3, 4, 5)
        entropies = [comp.entropy() for comp in self.mixture_model.components]
        ax5.bar(range(len(entropies)), entropies)
        ax5.set_title("Component Entropies")
        ax5.set_xlabel("Component")
        ax5.set_ylabel("Entropy")
        ax5.grid(True, alpha=0.3)

        # 5. Data assignment (if data provided)
        if data is not None:
            ax6 = plt.subplot(3, 4, 6)
            # Compute assignments
            responsibilities = np.zeros((len(data), self.mixture_model.n_components))
            for k in range(self.mixture_model.n_components):
                component = self.mixture_model.components[k]
                log_probs = component.log_pdf(data)
                responsibilities[:, k] = np.log(self.weights[k]) + log_probs
            assignments = np.argmax(responsibilities, axis=1)

            for k in range(self.mixture_model.n_components):
                mask = assignments == k
                if np.any(mask):
                    ax6.scatter(
                        data_2d[mask, 0],
                        data_2d[mask, 1],
                        c=[colors[k]],
                        alpha=0.6,
                        s=20,
                    )
            ax6.set_title("Data Assignments")
            ax6.set_xlabel("PC1")
            ax6.set_ylabel("PC2")
            ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Comprehensive visualization saved to {output_path}")

        return fig


def visualize_mixture_model(
    mixture_model: VonMisesFisherMixture,
    data: Optional[np.ndarray] = None,
    output_dir: str = "visualizations",
):
    """
    Create and save all visualizations for a mixture model.

    Args:
        mixture_model: Fitted von Mises-Fisher mixture model
        data: Data points to include in visualizations
        output_dir: Directory to save visualizations
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    visualizer = VonMisesFisherVisualizer(mixture_model)

    # Create individual plots
    print("Creating component parameters plot...")
    fig1 = visualizer.plot_component_parameters()
    fig1.savefig(
        os.path.join(output_dir, "component_parameters.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig1)

    print("Creating mean directions 2D plot...")
    fig2 = visualizer.plot_mean_directions_2d(data, method="pca")
    fig2.savefig(
        os.path.join(output_dir, "mean_directions_2d_pca.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig2)

    if mixture_model.dimension >= 3:
        print("Creating mean directions 3D plot...")
        fig3 = visualizer.plot_mean_directions_3d(data)
        fig3.savefig(
            os.path.join(output_dir, "mean_directions_3d.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig3)

    print("Creating similarity matrix plot...")
    fig4 = visualizer.plot_component_similarity_matrix()
    fig4.savefig(
        os.path.join(output_dir, "similarity_matrix.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig4)

    if data is not None:
        print("Creating data assignment plot...")
        fig5 = visualizer.plot_data_assignment(data, method="pca")
        fig5.savefig(
            os.path.join(output_dir, "data_assignments.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig5)

    print("Creating component entropy plot...")
    fig6 = visualizer.plot_component_entropy()
    fig6.savefig(
        os.path.join(output_dir, "component_entropies.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig6)

    print("Creating comprehensive visualization...")
    fig7 = visualizer.create_comprehensive_visualization(data)
    fig7.savefig(
        os.path.join(output_dir, "comprehensive_visualization.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig7)

    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Test visualization with synthetic data
    print("Testing von Mises-Fisher visualization...")

    from von_mises_fisher import VonMisesFisherMixture, VonMisesFisher

    # Create synthetic mixture model
    mixture = VonMisesFisherMixture(3, 3)
    mixture.add_component(np.array([1.0, 0.0, 0.0]), 5.0)
    mixture.add_component(np.array([0.0, 1.0, 0.0]), 3.0)
    mixture.add_component(np.array([0.0, 0.0, 1.0]), 2.0)

    # Generate sample data
    data = mixture.sample(200)

    # Create visualizations
    visualize_mixture_model(mixture, data, "test_visualizations")
    print("Visualization test completed!")
