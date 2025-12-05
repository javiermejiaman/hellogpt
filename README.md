<p align="center">
  <img src="docs/hellogpt-banner.jpg" alt="Project Banner">
</p>

# hellogpt
HelloGPT is a small-scale GPT-style model designed to showcase the building blocks of modern language models, including token embeddings, positional embeddings, self-attention, and feed-forward layers.

## Getting started
Python is required to run this project. It can be downloaded from [Python's website](https://www.python.org/downloads/) or [Anaconda](https://www.anaconda.com/download). The project was tested with Python 3.13.5, using Anaconda.

### Dependencies
Dependencies are documented in the file `requirements.txt`. All required dependencies can be installed with the command:

`pip install -r requirements.txt`

`Note: PyTorch is not included in the requirements.txt file but is required. You can install it following the instructions on the official PyTorch website, which provides a tool to generate the correct installation command for your system.`

### Running the project
If the previous steps completed successfully, run the main application with `python main.py`.

`Note: the command python could be different in your environment: py, python3.`

### Using the application

### Project structure
- main.py: Entry point of the project. Provides a user-friendly way to interact with the model and run training.
- data/: This folder stores the training data, the README.md inside contains more details.
- artifacts/: This folder stores the artifacts produced by the project.
  - `checkpoints/` will store the model's state.  
  - `tokenizer/` will store the trained tokenizer.
- src/: This folder stores the source code of the project.
- tests/: This folder stores the tests files. To execute the tests, run `pytest`, `pytest smoke` for smoke test or `pytest integration` for integration tests.

## Hardware requirements
Training AI/ML models today is very computationally intensive, even for small models like the one in this project. For this reason, it is recommended to use at least one GPU during training. Tests were carried out on an NVIDIA RTX 3060.

**Important:** Currently, only NVIDIA GPUs are supported by PyTorch, the ML/AI framework used in this project. PyTorch requires the CUDA toolkit to be installed on your system, which can be downloaded [here](https://developer.nvidia.com/cuda-toolkit).

## Model architecture:
- 1 -> Embeddings lookup and addition
- 2 -> Dropout
- 3 -> -> -> Transformers
- 4 -> Layer normalization
- 5 -> Head layer

#### Description:
- 1\) Encodes tokens and their position in the sequence.
- 2\) Randomly disables activations to prevent model overfitting.
- 3\) Applies self attention and feed-forward networks to enrich embeddings with contextual information.
- 4\) Stabilizes training and prevent gradient spikes.
- 5\) Linear projection mapping each token embedding to the vocabulaby logits.

## Transformer architecture:

#### Attention block
- CM -> Causal mask
- LN -> Layer normalization
- AT -> Multi-head attention
- RD -> Residual connections + dropout
  
#### Feed-forward block
- LN -> Layer normalization
- FF -> Feed-forward sub-layers
- RD -> Residual connections + dropout

#### Description:
- CM) Prevents tokens from attending to future positions.
- LN) Stabilizes training and prevent gradient spikes.
- AT) Enriches embeddings with contextual information.
- FF) Neural network applied independently to each token embedding to produce a refined representation.
- RD) Prevents gradient vanishing by letting it flow through the skip path. Drop out is also applied to disable activations and prevent overfitting.

## Training