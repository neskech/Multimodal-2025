# Von Mises-Fisher Mixture Model for CLIP Embeddings

This project implements a von Mises-Fisher mixture model for modeling CLIP embeddings on the unit sphere. The von Mises-Fisher distribution is particularly well-suited for this task since CLIP embeddings are normalized to unit length, making them naturally lie on the unit sphere.

## Features

- **CLIP Embedding Extraction**: Extract embeddings from text and images using OpenAI's CLIP model
- **Von Mises-Fisher Distribution**: Implementation of the von Mises-Fisher distribution on the unit sphere
- **EM Algorithm**: Expectation-Maximization algorithm for fitting mixture models
- **Comprehensive Visualization**: Multiple visualization tools for analyzing fitted models
- **Flexible Input**: Support for text files, image directories, or synthetic data generation

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The CLIP model will be automatically downloaded on first use.

## Quick Start

### Basic Example

Run the example script to see the system in action:

```bash
python example.py
```

This will:
- Generate sample CLIP embeddings
- Fit a 3-component von Mises-Fisher mixture model
- Create visualizations of the results

### Embedding Generation

Use the new `generate_embeddings.py` script to generate embeddings for different datasets:

#### COCO Dataset
```bash
python generate_embeddings.py --dataset coco --split train2017 --max_samples 1000
```

#### LAION Sample (text-only demonstration)
```bash
python generate_embeddings.py --dataset laion_sample
```

#### Custom Images
```bash
python generate_embeddings.py --dataset custom_images --custom_path /path/to/images
```

#### Custom Text File
```bash
python generate_embeddings.py --dataset custom_text --custom_path your_texts.txt
```

#### Different CLIP Models
```bash
# Using OpenCLIP with LAION pretrained models
python generate_embeddings.py --dataset coco --model_name ViT-L-14 --pretrained_name laion2b_s32b_b82k --use_openclip

# Using original CLIP models
python generate_embeddings.py --dataset coco --model_name ViT-L/14
```

#### Interactive Examples
```bash
python example_generate_embeddings.py
```

### Using Your Own Data with Main Script

#### With Text Data

Create a text file with one text per line:
```
a photo of a cat
a dog playing in the park
a beautiful sunset over mountains
```

Then run:
```bash
python main.py --text_file your_texts.txt --n_components 5
```

#### With Image Data

Place your images in a directory and run:
```bash
python main.py --image_dir /path/to/images --n_components 5
```

#### With Both Text and Images

```bash
python main.py --text_file texts.txt --image_dir /path/to/images --n_components 5
```

## Command Line Options

### Main Script (`main.py`)
- `--n_components`: Number of mixture components (default: 5)
- `--max_iterations`: Maximum EM iterations (default: 100)
- `--tolerance`: Convergence tolerance (default: 1e-6)
- `--random_state`: Random seed for reproducibility (default: 42)
- `--text_file`: Path to text file with one text per line
- `--image_dir`: Path to directory containing images
- `--n_samples`: Number of sample embeddings if no data provided (default: 1000)
- `--output_dir`: Output directory for results (default: results)
- `--clip_model`: CLIP model to use (default: ViT-B/32)
- `--test_split`: Fraction of data to use for testing (default: 0.2)

### Embedding Generation (`generate_embeddings.py`)
- `--dataset`: Dataset to use (choices: coco, laion_sample, custom_images, custom_text)
- `--data_dir`: Data directory (default: data)
- `--custom_path`: Path to custom images dir or text file
- `--split`: COCO dataset split (train2017, val2017)
- `--max_samples`: Maximum number of samples to process
- `--model_name`: CLIP model name (default: ViT-B/32)
- `--pretrained_name`: Pretrained checkpoint name (for OpenCLIP)
- `--use_openclip`: Use OpenCLIP instead of original CLIP
- `--device`: Device to use (cuda/cpu)
- `--cache_file`: Output cache file path
- `--force_recompute`: Force recompute even if cache exists
- `--include_images`: Include image embeddings (default: True)
- `--include_texts`: Include text embeddings (default: True)

## Output

The system generates several outputs:

1. **Model Parameters** (`model_parameters.json`): Fitted mixture model parameters
2. **Convergence Info** (`convergence_info.json`): EM algorithm convergence details
3. **Evaluation Metrics** (`evaluation_metrics.json`): Model performance metrics
4. **Convergence Plot** (`convergence_plot.png`): Log-likelihood vs iteration
5. **Visualizations** (`visualizations/` directory):
   - Component parameters (weights, concentrations)
   - Mean directions in 2D/3D
   - Component similarity matrix
   - Data assignments
   - Component entropies
   - Comprehensive overview

## Understanding the Results

### Component Parameters
- **Weights**: Relative importance of each component
- **Mean Directions**: Central directions of each component on the unit sphere
- **Concentrations**: How concentrated each component is (higher = more concentrated)

### Visualizations
- **Mean Directions**: Show where each component is centered in the embedding space
- **Similarity Matrix**: Shows how similar component directions are to each other
- **Data Assignments**: Shows which component each data point is most likely to belong to
- **Entropies**: Measure of uncertainty/spread for each component

## Mathematical Background

The von Mises-Fisher distribution on the unit sphere has the probability density function:

```
f(x) = C_d(k) * exp(k * μ^T * x)
```

Where:
- `μ` is the mean direction (unit vector)
- `k` is the concentration parameter (k ≥ 0)
- `C_d(k)` is the normalization constant
- `x` is a point on the unit sphere

The mixture model combines multiple von Mises-Fisher distributions:

```
P(x) = Σ_{i=1}^K π_i * vMF(x | μ_i, k_i)
```

Where `π_i` are the mixture weights.

## File Structure

- `clip_embeddings.py`: CLIP embedding extraction functionality
- `generate_embeddings.py`: **NEW** Comprehensive embedding generation for different datasets
- `example_generate_embeddings.py`: **NEW** Interactive examples for embedding generation  
- `von_mises_fisher.py`: Von Mises-Fisher distribution implementation
- `em_algorithm.py`: EM algorithm for fitting mixture models
- `visualization.py`: Visualization tools
- `main.py`: Main script with command-line interface
- `example.py`: Simple example script
- `cardelph_projection.py`: Existing projection visualization script
- `requirements.txt`: Python dependencies

## Advanced Usage

### Custom CLIP Models

You can use different CLIP model variants:

```bash
python main.py --clip_model ViT-L/14 --n_components 10
```

Available models: `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `RN50`, `RN101`, `RN50x4`, `RN50x16`, `RN50x64`

### Programmatic Usage

```python
from clip_embeddings import CLIPEmbeddingExtractor
from em_algorithm import fit_von_mises_fisher_mixture
from visualization import visualize_mixture_model

# Extract embeddings
extractor = CLIPEmbeddingExtractor()
texts = ["a cat", "a dog", "a bird"]
embeddings = extractor.extract_text_embeddings(texts)

# Fit mixture model
mixture_model, convergence_info = fit_von_mises_fisher_mixture(
    embeddings, n_components=3
)

# Visualize results
visualize_mixture_model(mixture_model, embeddings)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use a smaller CLIP model or reduce batch size
2. **Slow convergence**: Increase `max_iterations` or adjust `tolerance`
3. **Poor fit**: Try different number of components or check data quality

### Performance Tips

- Use GPU if available (automatically detected)
- For large datasets, consider using a smaller CLIP model
- The EM algorithm can be slow for high-dimensional embeddings

## References

- Banerjee, A., Dhillon, I. S., Ghosh, J., & Sra, S. (2005). Clustering on the unit hypersphere using von Mises-Fisher distributions. JMLR.
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. ICML.

## License

This project is open source. Please check the individual dependencies for their respective licenses.
