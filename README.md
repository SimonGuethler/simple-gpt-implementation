# Simple GPT Implementation

A simple example for implementing a Generative Pre-trained Transformer with pytorch.
The entire code for the model is taken from [Andrej Karpathy's video course Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=kCc8FmEb1nY), forked from https://github.com/karpathy/ng-video-lecture
The original contents of the README.md file are preserved at the bottom of this document.

## Setup for NVIDIA GPUs

    conda create -n simple-gpt python=3.9
    conda activate simple-gpt
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

## Training the network

Files are loaded from /input. Each file is split into 90% training data and 10% test data.

To start training, execute train.py

    python train.py

The resulting model will be saved to /models/model/model-last.pt after every evaluation.

## Generating something from the network

To perform sentence completion using the trained network, run

    python generate.py


# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.


### License

MIT
