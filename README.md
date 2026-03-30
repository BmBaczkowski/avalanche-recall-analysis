# Analysis Pipeline of Free Recall Data

## Overview

A natural language processing pipeline for analyzing free recall of avalanche-related text.

## Key Features

- PyTorch-accelerated text analysis
- Sentence embedding with Sentence-BERT (SBERT)
- Text generation using GPT-4o
- Machine learning classification
- Embedding visualization and analysis

## Pipeline Stages

1. **Sentence Generation**
   - Uses GPT-4o to generate related sentences about avalanches
   - Supports both simplified and detailed processing modes
   - Segments input text and generates multiple responses

2. **Embedding Generation**
   - Converts generated sentences into numerical vector representations
   - Utilizes MPNet model from Sentence-BERT
   - Leverages PyTorch for efficient tensor computations

3. **Classification and Visualization**
   - Applies random forest classifiers
   - Generates probability heatmaps
   - Visualizes sentence recall and topic probabilities

## Technologies

- PyTorch
- Sentence-BERT (SBERT)
- GPT-4o
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Project Structure

- `sentences_gpt-4o/`: Scripts for sentence generation and embedding
- `recall-analysis/`: Classification and analysis scripts
- `utilities_py/`: Utility functions and helper modules

## Requirements

- Python 3.8+
- PyTorch
- Sentence-BERT
- OpenAI API (for GPT-4o)
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Usage

Refer to the Makefiles in each directory for specific execution commands. The project uses a Docker-based workflow with PyTorch containers.

## Performance Notes

The pipeline is optimized for computational efficiency, utilizing PyTorch's parallel processing and tensor computation capabilities.

## License

MIT license
