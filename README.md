# Simple GPT Implementation

A simple example for implementing a Generative Pre-trained Transformer with pytorch.
The entire code for the model is taken from [Andrej Karpathy's video course Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=kCc8FmEb1nY), forked from https://github.com/karpathy/ng-video-lecture  
The original contents of the README.md file are preserved at the bottom of this document.

This repository differs by the following small changes:
- Support loading multiple files
- Store model to disk during and after training
- Put code used for training and inference into separate scripts
- Partially added support for training at fp16 percision for decreased memory usage ([source](https://gist.github.com/ajbrock/075c0ca4036dc4d8581990a6e76e07a3))

Many improvements could be made to saving/loading the models and parameters, but as this is just research project, code is meant to be kept simple. Also there are already a dozen highly optimized implementations of the demonstrated techniques.

## Setup for NVIDIA GPUs

    conda create -n simple-gpt python=3.9
    conda activate simple-gpt
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

Also, Cuda 11.7 or higher must be installed and the CUDA_PATH environment variable must point to the corresponding directory, e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

Unfortunately, this code base only supports a single GPU for training and inference.

## Training the network

Files are loaded from /input. Each file is split into 90% training data and 10% test data.

To start training, execute train.py

    python train.py

The resulting model will be saved to /models/model/model-last.pt after every evaluation.
Additionally, the evaluation results will be used to save the model to model-best-train.pt and model-best-val.pt.

## Generating something from the network

To perform sentence completion using the trained network, run

    python generate.py

# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.


### License

MIT
