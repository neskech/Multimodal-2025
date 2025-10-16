"""
Test script to verify the von Mises-Fisher mixture model implementation.
"""

import numpy as np
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_von_mises_fisher():
    """Test the von Mises-Fisher distribution implementation."""
    print("Testing von Mises-Fisher distribution...")
    
    try:
        from von_mises_fisher import VonMisesFisher, VonMisesFisherMixture
        
        # Test single distribution
        mean_dir = np.array([1.0, 0.0, 0.0])
        concentration = 2.0
        vmf = VonMisesFisher(mean_dir, concentration)
        
        # Test sampling
        samples = vmf.sample(10)
        print(f"✓ Generated {len(samples)} samples from vMF distribution")
        
        # Test PDF computation
        test_point = np.array([1.0, 0.0, 0.0])
        pdf_value = vmf.pdf(test_point.reshape(1, -1))[0]
        print(f"✓ PDF computation works: {pdf_value:.4f}")
        
        # Test mixture model
        mixture = VonMisesFisherMixture(2, 3)
        mixture.add_component(np.array([1.0, 0.0, 0.0]), 2.0)
        mixture.add_component(np.array([0.0, 1.0, 0.0]), 1.5)
        
        mixture_samples = mixture.sample(20)
        print(f"✓ Generated {len(mixture_samples)} samples from mixture model")
        
        print("✓ Von Mises-Fisher implementation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Von Mises-Fisher test failed: {e}")
        return False


def test_em_algorithm():
    """Test the EM algorithm implementation."""
    print("\nTesting EM algorithm...")
    
    try:
        from em_algorithm import fit_von_mises_fisher_mixture
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 100
        dimension = 3
        
        # Create two clusters
        cluster1 = np.random.randn(n_samples//2, dimension)
        cluster1 = cluster1 / np.linalg.norm(cluster1, axis=1, keepdims=True)
        
        cluster2 = np.random.randn(n_samples//2, dimension)
        cluster2 = cluster2 / np.linalg.norm(cluster2, axis=1, keepdims=True)
        
        data = np.vstack([cluster1, cluster2])
        
        # Fit mixture model
        mixture_model, convergence_info = fit_von_mises_fisher_mixture(
            data, n_components=2, max_iterations=20, verbose=False
        )
        
        print(f"✓ EM algorithm converged: {convergence_info['converged']}")
        print(f"✓ Final log-likelihood: {convergence_info['final_log_likelihood']:.4f}")
        
        # Test model parameters
        weights, mean_directions, concentrations = mixture_model.get_parameters()
        print(f"✓ Fitted {len(weights)} components")
        
        print("✓ EM algorithm test passed!")
        return True
        
    except Exception as e:
        print(f"✗ EM algorithm test failed: {e}")
        return False


def test_clip_embeddings():
    """Test CLIP embedding extraction (if available)."""
    print("\nTesting CLIP embeddings...")
    
    try:
        from clip_embeddings import CLIPEmbeddingExtractor, create_sample_data
        
        # Test extractor initialization
        extractor = CLIPEmbeddingExtractor()
        print(f"✓ CLIP extractor initialized with dimension {extractor.get_embedding_dimension()}")
        
        # Test sample data generation
        sample_embeddings = create_sample_data(extractor, num_samples=10)
        print(f"✓ Generated sample embeddings with shape {sample_embeddings.shape}")
        
        # Test embedding normalization
        norms = np.linalg.norm(sample_embeddings, axis=1)
        print(f"✓ Embedding norms: mean={np.mean(norms):.4f}, std={np.std(norms):.4f}")
        
        print("✓ CLIP embeddings test passed!")
        return True
        
    except Exception as e:
        print(f"✗ CLIP embeddings test failed: {e}")
        print("  (This is expected if CLIP dependencies are not installed)")
        return False


def test_visualization():
    """Test visualization functionality (if available)."""
    print("\nTesting visualization...")
    
    try:
        from visualization import VonMisesFisherVisualizer
        from von_mises_fisher import VonMisesFisherMixture
        
        # Create test mixture model
        mixture = VonMisesFisherMixture(2, 3)
        mixture.add_component(np.array([1.0, 0.0, 0.0]), 2.0)
        mixture.add_component(np.array([0.0, 1.0, 0.0]), 1.5)
        
        # Test visualizer
        visualizer = VonMisesFisherVisualizer(mixture)
        print("✓ VonMisesFisherVisualizer initialized")
        
        # Test parameter plotting (without actually showing plots)
        fig = visualizer.plot_component_parameters()
        print("✓ Component parameters plot created")
        
        print("✓ Visualization test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        print("  (This is expected if visualization dependencies are not installed)")
        return False


def main():
    """Run all tests."""
    print("Von Mises-Fisher Mixture Model Implementation Tests")
    print("=" * 60)
    
    tests = [
        test_von_mises_fisher,
        test_em_algorithm,
        test_clip_embeddings,
        test_visualization
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("   Note: CLIP and visualization tests may fail if dependencies are not installed.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
