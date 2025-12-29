# BERT Fine-tuning for Masked Language Modeling

A complete implementation for fine-tuning BERT (or DistilBERT) models using Masked Language Modeling (MLM) on custom datasets. This project includes optimizations for faster training, better memory efficiency, and proper evaluation practices.

## Features

- **Optimized Training**: Mixed precision (FP16), parallel data loading, and efficient chunking
- **Dynamic Masking**: Different masks each epoch during training for better generalization
- **Consistent Evaluation**: Fixed masks for evaluation to ensure fair comparison across epochs
- **Early Stopping**: Automatic stopping when model stops improving
- **Checkpointing**: Saves best model based on perplexity
- **Multi-GPU Support**: Works seamlessly with multiple GPUs via HuggingFace Accelerate

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch
- Transformers library
- Datasets library

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd bert_finetuning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install transformers datasets accelerate torch
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook optimized_mlm_finetuning.ipynb
```

2. Run all cells sequentially. The notebook will:
   - Check GPU availability
   - Install required packages
   - Load and preprocess the dataset
   - Fine-tune the model
   - Save checkpoints and training history

## Configuration

All hyperparameters are configured in the `Config` class within the notebook. Key settings:

- `model_name`: Pre-trained model to use (default: `distilbert-base-uncased`)
- `dataset_name`: Dataset to fine-tune on (default: `imdb`)
- `num_epochs`: Number of training epochs (default: 10)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 5e-5)
- `mlm_probability`: Masking probability (default: 0.15)

## Output

The training process creates:
- `mlm_checkpoints/`: Directory containing model checkpoints
- `training_history.json`: JSON file with training metrics (loss, perplexity, learning rate)

## Key Design Decisions

- **Training**: Uses dynamic masking (different masks each epoch) for better generalization
- **Evaluation**: Uses fixed masks (same masks each epoch) for fair comparison
- **Memory Efficiency**: Dynamic masking during training avoids storing multiple masked versions
- **Performance**: Mixed precision training and optimized data loading for faster training

## Performance Optimizations

- Mixed precision (FP16) for ~2x speedup
- Parallel data loading with multiple workers
- Efficient text chunking strategy
- Pin memory for faster GPU transfer
- Gradient accumulation support

## License

MIT License - see LICENSE file for details

## Contributing

Feel free to open issues or submit pull requests if you find bugs or have suggestions for improvements.

