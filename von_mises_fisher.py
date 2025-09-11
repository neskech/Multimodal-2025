"""
Von Mises-Fisher distribution implementation for mixture modeling on the unit sphere.
"""

import numpy as np
import torch
from scipy.special import iv, gamma
from typing import Tuple, Optional
import warnings


class VonMisesFisher:
    """
    Von Mises-Fisher distribution on the unit sphere.
    
    The probability density function is:
    f(x) = C_d(k) * exp(k * μ^T * x)
    
    where:
    - μ is the mean direction (unit vector)
    - k is the concentration parameter (k >= 0)
    - C_d(k) is the normalization constant
    """
    
    def __init__(self, mean_direction: np.ndarray, concentration: float):
        """
        Initialize von Mises-Fisher distribution.
        
        Args:
            mean_direction: Mean direction vector (will be normalized)
            concentration: Concentration parameter k (must be >= 0)
        """
        self.mean_direction = self._normalize(mean_direction)
        self.concentration = max(0.0, concentration)  # Ensure non-negative
        self.dimension = len(self.mean_direction)
        
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")
        return vector / norm
    
    def log_normalization_constant(self) -> float:
        """
        Compute log of normalization constant C_d(k).
        
        C_d(k) = k^(d/2-1) / ((2π)^(d/2) * I_{d/2-1}(k))
        """
        d = self.dimension
        k = self.concentration
        
        if k == 0:
            # Uniform distribution on sphere
            return -np.log(2 * np.pi**(d/2) / gamma(d/2))
        
        # Use scipy's modified Bessel function
        bessel_order = d/2 - 1
        log_bessel = np.log(iv(bessel_order, k))
        
        log_c = (d/2 - 1) * np.log(k) - (d/2) * np.log(2 * np.pi) - log_bessel
        
        return log_c
    
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log probability density function.
        
        Args:
            x: Points on unit sphere (shape: (n_samples, dimension))
            
        Returns:
            Log probabilities (shape: (n_samples,))
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Ensure x is on unit sphere
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        if not np.allclose(x_norm, 1.0, atol=1e-6):
            warnings.warn("Input points are not on unit sphere, normalizing...")
            x = x / x_norm
        
        # Compute dot products
        dot_products = np.dot(x, self.mean_direction)
        
        # Log PDF: log(C_d(k)) + k * μ^T * x
        log_c = self.log_normalization_constant()
        log_probs = log_c + self.concentration * dot_products
        
        return log_probs
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute probability density function."""
        return np.exp(self.log_pdf(x))
    
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample from von Mises-Fisher distribution using rejection sampling.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Samples on unit sphere (shape: (n_samples, dimension))
        """
        if self.concentration == 0:
            # Uniform distribution on sphere
            return self._sample_uniform_sphere(n_samples)
        
        # For high concentration, use a more efficient sampling method
        if self.concentration > 10:
            return self._sample_high_concentration(n_samples)
        
        samples = []
        max_attempts = n_samples * 50  # Increase max attempts
        attempts = 0
        
        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Generate candidate from uniform distribution
            candidate = self._sample_uniform_sphere(1)[0]
            
            # Compute acceptance probability
            # For vMF, we need to use the correct envelope function
            dot_product = np.dot(candidate, self.mean_direction)
            
            # Use a more appropriate envelope for rejection sampling
            # The envelope should be exp(k * (μ^T * x - 1)) / exp(k * (1 - 1)) = exp(k * (μ^T * x - 1))
            # But we need to normalize this properly
            acceptance_prob = np.exp(self.concentration * (dot_product - 1))
            
            # Ensure acceptance probability is valid
            acceptance_prob = min(1.0, max(0.0, acceptance_prob))
            
            if np.random.random() < acceptance_prob:
                samples.append(candidate)
        
        # If we still don't have enough samples, use fallback method
        if len(samples) < n_samples:
            remaining = n_samples - len(samples)
            fallback_samples = self._sample_fallback(remaining)
            samples.extend(fallback_samples)
            
        if len(samples) < n_samples:
            warnings.warn(f"Only generated {len(samples)} samples out of {n_samples} requested")
        
        return np.array(samples[:n_samples])  # Ensure we don't return more than requested
    
    def _sample_uniform_sphere(self, n_samples: int) -> np.ndarray:
        """Sample uniformly from unit sphere."""
        # Generate random points in d-dimensional space
        points = np.random.randn(n_samples, self.dimension)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms
    
    def _sample_high_concentration(self, n_samples: int) -> np.ndarray:
        """
        Sample from vMF distribution for high concentration parameters.
        Uses a more efficient method based on the fact that high concentration
        means the distribution is very concentrated around the mean direction.
        """
        # For high concentration, sample from a normal distribution
        # and project to the sphere
        samples = []
        
        for _ in range(n_samples):
            # Generate a small perturbation around the mean direction
            perturbation = np.random.randn(self.dimension) * (1.0 / np.sqrt(self.concentration))
            candidate = self.mean_direction + perturbation
            
            # Project back to unit sphere
            candidate = candidate / np.linalg.norm(candidate)
            samples.append(candidate)
        
        return np.array(samples)
    
    def _sample_fallback(self, n_samples: int) -> np.ndarray:
        """
        Fallback sampling method when rejection sampling fails.
        Uses a simple approach: sample uniformly and weight by the PDF.
        """
        # Generate many uniform samples
        candidates = self._sample_uniform_sphere(n_samples * 10)
        
        # Compute PDF values
        pdf_values = self.pdf(candidates)
        
        # Normalize to get probabilities
        probabilities = pdf_values / np.sum(pdf_values)
        
        # Sample according to these probabilities
        indices = np.random.choice(
            len(candidates), 
            size=n_samples, 
            p=probabilities,
            replace=True
        )
        
        return candidates[indices]
    
    def expected_dot_product(self) -> float:
        """
        Compute expected value of μ^T * X.
        
        For vMF distribution: E[μ^T * X] = I_{d/2}(k) / I_{d/2-1}(k)
        """
        if self.concentration == 0:
            return 0.0
        
        d = self.dimension
        k = self.concentration
        
        bessel_numerator = iv(d/2, k)
        bessel_denominator = iv(d/2 - 1, k)
        
        if bessel_denominator == 0:
            return 1.0  # Limit as k -> infinity
        
        return bessel_numerator / bessel_denominator
    
    def entropy(self) -> float:
        """
        Compute entropy of the distribution.
        
        H = -log(C_d(k)) - k * E[μ^T * X]
        """
        log_c = self.log_normalization_constant()
        expected_dot = self.expected_dot_product()
        
        return -log_c - self.concentration * expected_dot


class VonMisesFisherMixture:
    """
    Mixture of von Mises-Fisher distributions.
    
    P(x) = Σ_{i=1}^K π_i * vMF(x | μ_i, k_i)
    """
    
    def __init__(self, n_components: int, dimension: int):
        """
        Initialize mixture model.
        
        Args:
            n_components: Number of mixture components
            dimension: Dimension of the embedding space
        """
        self.n_components = n_components
        self.dimension = dimension
        self.components = []
        self.weights = np.ones(n_components) / n_components  # Uniform weights initially
        
    def add_component(self, mean_direction: np.ndarray, concentration: float):
        """Add a component to the mixture."""
        component = VonMisesFisher(mean_direction, concentration)
        self.components.append(component)
    
    def log_likelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log likelihood for each data point.
        
        Args:
            x: Data points on unit sphere (shape: (n_samples, dimension))
            
        Returns:
            Log likelihoods (shape: (n_samples,))
        """
        if len(self.components) == 0:
            raise ValueError("No components added to mixture")
        
        n_samples = x.shape[0]
        log_likelihoods = np.zeros(n_samples)
        
        for i, component in enumerate(self.components):
            component_log_probs = component.log_pdf(x)
            log_likelihoods += self.weights[i] * np.exp(component_log_probs)
        
        return np.log(log_likelihoods + 1e-10)  # Add small epsilon for numerical stability
    
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute log probability density function."""
        return self.log_likelihood(x)
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Compute probability density function."""
        return np.exp(self.log_pdf(x))
    
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample from the mixture model.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Samples on unit sphere (shape: (n_samples, dimension))
        """
        if len(self.components) == 0:
            raise ValueError("No components added to mixture")
        
        # Sample component assignments
        component_indices = np.random.choice(
            len(self.components), 
            size=n_samples, 
            p=self.weights
        )
        
        samples = []
        for idx in component_indices:
            component_sample = self.components[idx].sample(1)[0]
            samples.append(component_sample)
        
        return np.array(samples)
    
    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mixture parameters.
        
        Returns:
            Tuple of (weights, mean_directions, concentrations)
        """
        weights = self.weights.copy()
        mean_directions = np.array([comp.mean_direction for comp in self.components])
        concentrations = np.array([comp.concentration for comp in self.components])
        
        return weights, mean_directions, concentrations
    
    def set_parameters(self, weights: np.ndarray, mean_directions: np.ndarray, 
                      concentrations: np.ndarray):
        """
        Set mixture parameters.
        
        Args:
            weights: Component weights (must sum to 1)
            mean_directions: Mean directions for each component
            concentrations: Concentration parameters for each component
        """
        if len(weights) != self.n_components:
            raise ValueError("Number of weights must match number of components")
        
        # Normalize weights
        self.weights = weights / np.sum(weights)
        
        # Update components
        self.components = []
        for i in range(self.n_components):
            self.add_component(mean_directions[i], concentrations[i])


if __name__ == "__main__":
    # Test von Mises-Fisher distribution
    print("Testing von Mises-Fisher distribution...")
    
    # Create a test distribution
    mean_dir = np.array([1.0, 0.0, 0.0])
    concentration = 2.0
    vmf = VonMisesFisher(mean_dir, concentration)
    
    print(f"Mean direction: {vmf.mean_direction}")
    print(f"Concentration: {vmf.concentration}")
    print(f"Log normalization constant: {vmf.log_normalization_constant()}")
    print(f"Expected dot product: {vmf.expected_dot_product()}")
    print(f"Entropy: {vmf.entropy()}")
    
    # Test sampling
    samples = vmf.sample(100)
    print(f"Generated {len(samples)} samples")
    print(f"Sample mean direction: {np.mean(samples, axis=0)}")
    
    # Test mixture model
    print("\nTesting von Mises-Fisher mixture...")
    mixture = VonMisesFisherMixture(2, 3)
    mixture.add_component(np.array([1.0, 0.0, 0.0]), 2.0)
    mixture.add_component(np.array([0.0, 1.0, 0.0]), 1.5)
    
    test_point = np.array([1.0, 0.0, 0.0])
    print(f"Log PDF at test point: {mixture.log_pdf(test_point.reshape(1, -1))}")
    
    mixture_samples = mixture.sample(50)
    print(f"Generated {len(mixture_samples)} mixture samples")
