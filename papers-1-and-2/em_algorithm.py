"""
Expectation-Maximization algorithm for fitting von Mises-Fisher mixture models.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import iv, ivp
from typing import Tuple, Optional, List
import warnings

from von_mises_fisher import VonMisesFisher, VonMisesFisherMixture


class VonMisesFisherEM:
    """
    Expectation-Maximization algorithm for von Mises-Fisher mixture models.
    """
    
    def __init__(self, n_components: int, max_iterations: int = 100, 
                 tolerance: float = 1e-6, random_state: Optional[int] = None):
        """
        Initialize EM algorithm.
        
        Args:
            n_components: Number of mixture components
            max_iterations: Maximum number of EM iterations
            tolerance: Convergence tolerance
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.mixture_model = None
        self.log_likelihood_history = []
        self.converged = False
        
    def _initialize_parameters(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize mixture parameters using k-means-like approach.
        
        Args:
            data: Data points on unit sphere (shape: (n_samples, dimension))
            
        Returns:
            Tuple of (weights, mean_directions, concentrations)
        """
        n_samples, dimension = data.shape
        
        # Initialize weights uniformly
        weights = np.ones(self.n_components) / self.n_components
        
        # Initialize mean directions using k-means++ approach
        mean_directions = np.zeros((self.n_components, dimension))
        
        # First center: random data point
        first_idx = np.random.randint(n_samples)
        mean_directions[0] = data[first_idx]
        
        # Subsequent centers: choose points far from existing centers
        for k in range(1, self.n_components):
            distances = np.zeros(n_samples)
            for i in range(n_samples):
                # Distance to closest existing center
                min_dist = float('inf')
                for j in range(k):
                    dist = 1 - np.dot(data[i], mean_directions[j])  # Cosine distance
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist
            
            # Ensure distances are non-negative and add small epsilon to avoid zeros
            distances = np.maximum(distances, 1e-10)
            
            # Choose point with maximum minimum distance
            probabilities = distances / np.sum(distances)
            new_idx = np.random.choice(n_samples, p=probabilities)
            mean_directions[k] = data[new_idx]
        
        # Normalize mean directions
        for k in range(self.n_components):
            mean_directions[k] = mean_directions[k] / np.linalg.norm(mean_directions[k])
        
        # Initialize concentrations with moderate values
        concentrations = np.ones(self.n_components) * 1.0
        
        return weights, mean_directions, concentrations
    
    def _e_step(self, data: np.ndarray, weights: np.ndarray, 
                mean_directions: np.ndarray, concentrations: np.ndarray) -> np.ndarray:
        """
        Expectation step: compute posterior probabilities.
        
        Args:
            data: Data points (shape: (n_samples, dimension))
            weights: Component weights
            mean_directions: Mean directions for each component
            concentrations: Concentration parameters
            
        Returns:
            Posterior probabilities (shape: (n_samples, n_components))
        """
        n_samples = data.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Create temporary component
            component = VonMisesFisher(mean_directions[k], concentrations[k])
            log_probs = component.log_pdf(data)
            responsibilities[:, k] = np.log(weights[k]) + log_probs
        
        # Normalize to get probabilities (log-sum-exp trick)
        max_log_probs = np.max(responsibilities, axis=1, keepdims=True)
        responsibilities = responsibilities - max_log_probs
        responsibilities = np.exp(responsibilities)
        
        # Normalize rows to sum to 1
        row_sums = np.sum(responsibilities, axis=1, keepdims=True)
        responsibilities = responsibilities / row_sums
        
        return responsibilities
    
    def _m_step_weights(self, responsibilities: np.ndarray) -> np.ndarray:
        """Update component weights."""
        return np.mean(responsibilities, axis=0)
    
    def _m_step_mean_directions(self, data: np.ndarray, responsibilities: np.ndarray) -> np.ndarray:
        """Update mean directions."""
        n_samples, dimension = data.shape
        mean_directions = np.zeros((self.n_components, dimension))
        
        for k in range(self.n_components):
            # Weighted sum of data points
            weighted_sum = np.sum(responsibilities[:, k:k+1] * data, axis=0)
            # Normalize to get new mean direction
            norm = np.linalg.norm(weighted_sum)
            if norm > 1e-10:
                mean_directions[k] = weighted_sum / norm
            else:
                # Fallback to previous mean direction or random
                mean_directions[k] = np.random.randn(dimension)
                mean_directions[k] = mean_directions[k] / np.linalg.norm(mean_directions[k])
        
        return mean_directions
    
    def _m_step_concentrations(self, data: np.ndarray, responsibilities: np.ndarray,
                              mean_directions: np.ndarray) -> np.ndarray:
        """Update concentration parameters using maximum likelihood estimation."""
        n_samples, dimension = data.shape
        concentrations = np.zeros(self.n_components)
        
        for k in range(self.n_components):
            # Compute weighted average of dot products
            dot_products = np.dot(data, mean_directions[k])
            weighted_dot_product = np.sum(responsibilities[:, k] * dot_products) / np.sum(responsibilities[:, k])
            
            # Solve for concentration parameter
            concentration = self._solve_concentration(weighted_dot_product, dimension)
            concentrations[k] = concentration
        
        return concentrations
    
    def _solve_concentration(self, r_bar: float, dimension: int) -> float:
        """
        Solve for concentration parameter given mean resultant length.
        
        We need to solve: r_bar = I_{d/2}(k) / I_{d/2-1}(k)
        """
        if r_bar <= 0:
            return 0.0
        if r_bar >= 1:
            return 100.0  # Large concentration
        
        # Use numerical optimization to solve the equation
        def objective(k):
            if k <= 0:
                return float('inf')
            try:
                bessel_numerator = iv(dimension/2, k)
                bessel_denominator = iv(dimension/2 - 1, k)
                if bessel_denominator == 0:
                    return float('inf')
                predicted_r = bessel_numerator / bessel_denominator
                return (predicted_r - r_bar)**2
            except:
                return float('inf')
        
        # Use bounded optimization
        result = minimize_scalar(objective, bounds=(0.001, 100), method='bounded')
        
        if result.success:
            return result.x
        else:
            # Fallback: use approximation
            return self._approximate_concentration(r_bar, dimension)
    
    def _approximate_concentration(self, r_bar: float, dimension: int) -> float:
        """
        Approximate concentration parameter using analytical approximation.
        
        For high dimensions, we can use: k ≈ (d-1) * r_bar / (1 - r_bar)
        """
        if r_bar >= 0.99:
            return 100.0
        
        # Banerjee et al. approximation
        k = r_bar * (dimension - r_bar**2) / (1 - r_bar**2)
        return max(0.001, k)
    
    def fit(self, data: np.ndarray, verbose: bool = True) -> VonMisesFisherMixture:
        """
        Fit von Mises-Fisher mixture model to data.
        
        Args:
            data: Data points on unit sphere (shape: (n_samples, dimension))
            verbose: Whether to print progress
            
        Returns:
            Fitted mixture model
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2D array")
        
        n_samples, dimension = data.shape
        
        # Ensure data is on unit sphere
        norms = np.linalg.norm(data, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            warnings.warn("Data points are not on unit sphere, normalizing...")
            data = data / norms[:, np.newaxis]
        
        # Initialize parameters
        weights, mean_directions, concentrations = self._initialize_parameters(data)
        
        if verbose:
            print(f"Starting EM algorithm with {self.n_components} components...")
            print(f"Data shape: {data.shape}")
        
        # EM iterations
        for iteration in range(self.max_iterations):
            # E-step
            responsibilities = self._e_step(data, weights, mean_directions, concentrations)
            
            # M-step
            new_weights = self._m_step_weights(responsibilities)
            new_mean_directions = self._m_step_mean_directions(data, responsibilities)
            new_concentrations = self._m_step_concentrations(data, responsibilities, new_mean_directions)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(data, new_weights, new_mean_directions, new_concentrations)
            self.log_likelihood_history.append(log_likelihood)
            
            # Check convergence
            if iteration > 0:
                likelihood_change = log_likelihood - self.log_likelihood_history[-2]
                if abs(likelihood_change) < self.tolerance:
                    self.converged = True
                    if verbose:
                        print(f"Converged after {iteration + 1} iterations")
                    break
            
            # Update parameters
            weights = new_weights
            mean_directions = new_mean_directions
            concentrations = new_concentrations
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}: Log-likelihood = {log_likelihood:.4f}")
        
        if not self.converged and verbose:
            print(f"Did not converge after {self.max_iterations} iterations")
        
        # Create final mixture model
        self.mixture_model = VonMisesFisherMixture(self.n_components, dimension)
        self.mixture_model.set_parameters(weights, mean_directions, concentrations)
        
        return self.mixture_model
    
    def _compute_log_likelihood(self, data: np.ndarray, weights: np.ndarray,
                               mean_directions: np.ndarray, concentrations: np.ndarray) -> float:
        """Compute log-likelihood of data under current parameters."""
        n_samples = data.shape[0]
        log_likelihood = 0.0
        
        for i in range(n_samples):
            point_likelihood = 0.0
            for k in range(self.n_components):
                component = VonMisesFisher(mean_directions[k], concentrations[k])
                component_log_prob = component.log_pdf(data[i:i+1])[0]
                point_likelihood += weights[k] * np.exp(component_log_prob)
            log_likelihood += np.log(point_likelihood + 1e-10)
        
        return log_likelihood
    
    def get_convergence_info(self) -> dict:
        """Get information about convergence."""
        return {
            'converged': self.converged,
            'n_iterations': len(self.log_likelihood_history),
            'final_log_likelihood': self.log_likelihood_history[-1] if self.log_likelihood_history else None,
            'log_likelihood_history': self.log_likelihood_history
        }


def fit_von_mises_fisher_mixture(data: np.ndarray, n_components: int, 
                                max_iterations: int = 100, tolerance: float = 1e-6,
                                random_state: Optional[int] = None, verbose: bool = True) -> Tuple[VonMisesFisherMixture, dict]:
    """
    Convenience function to fit von Mises-Fisher mixture model.
    
    Args:
        data: Data points on unit sphere
        n_components: Number of mixture components
        max_iterations: Maximum EM iterations
        tolerance: Convergence tolerance
        random_state: Random seed
        verbose: Whether to print progress
        
    Returns:
        Tuple of (fitted_mixture_model, convergence_info)
    """
    em = VonMisesFisherEM(n_components, max_iterations, tolerance, random_state)
    mixture_model = em.fit(data, verbose)
    convergence_info = em.get_convergence_info()
    
    return mixture_model, convergence_info


if __name__ == "__main__":
    # Test EM algorithm with synthetic data
    print("Testing von Mises-Fisher EM algorithm...")
    
    # Generate synthetic data from two vMF distributions
    np.random.seed(42)
    
    # Component 1: concentrated around [1, 0, 0]
    vmf1 = VonMisesFisher(np.array([1.0, 0.0, 0.0]), 5.0)
    samples1 = vmf1.sample(100)
    
    # Component 2: concentrated around [0, 1, 0]
    vmf2 = VonMisesFisher(np.array([0.0, 1.0, 0.0]), 3.0)
    samples2 = vmf2.sample(100)
    
    # Combine data
    data = np.vstack([samples1, samples2])
    
    print(f"Generated synthetic data with shape: {data.shape}")
    
    # Fit mixture model
    mixture_model, convergence_info = fit_von_mises_fisher_mixture(
        data, n_components=2, verbose=True
    )
    
    print(f"\nConvergence info: {convergence_info}")
    
    # Get fitted parameters
    weights, mean_directions, concentrations = mixture_model.get_parameters()
    print(f"\nFitted parameters:")
    print(f"Weights: {weights}")
    print(f"Mean directions:\n{mean_directions}")
    print(f"Concentrations: {concentrations}")
