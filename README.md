# Simple GPT Implementation

A simple example for implementing a Generative Pre-trained Transformer with pytorch.
The entire code for the model is taken from [Andrej Karpathy's video course Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=kCc8FmEb1nY), forked from https://github.com/karpathy/ng-video-lecture  
The original contents of the README.md file are preserved at the bottom of this document.

This repository differs from the original in the following ways:
- Support loading multiple files
- Store model to disk during and after training
- Put code used for training and inference into separate scripts
- Partially added support for training at fp16 percision for decreased memory usage ([source](https://gist.github.com/ajbrock/075c0ca4036dc4d8581990a6e76e07a3))
- Added code to filter input data by removing characters that don't occurr very often (e.g. chinese characters from english wikipedia dumps)

Many improvements could be made to saving/loading the models, data, filters and parameters, but as this is just a research project, the code is meant to be kept simple. Also there exist already a dozen highly optimized implementations of the demonstrated techniques.

## Setup for NVIDIA GPUs

    conda create -n simple-gpt python=3.9
    conda activate simple-gpt
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

Also, Cuda 11.7 or higher must be installed and the CUDA_PATH environment variable must point to the corresponding directory, e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`

## Training the network

Files are loaded from /input. Each file is split into 90% training data and 10% test data.

To start training, execute train.py

    python train.py

The resulting model will be saved to /models/model/model-last.pt after every evaluation.  
Additionally, the evaluation results will be used to save the model to model-best-train.pt and model-best-val.pt.

## Generating something from the network

To perform sentence completion using the trained network, run

    python generate.py

# Finding data to train on

The included input text files contain about 1MB of Shakespeare's words.  

[This repository](https://github.com/TheMcSebi/wikipedia-top-corpus) contains a few preprocessed text-files from relatively recent wikipedia dumps (01/23), all in UTF-8. To keep the network at a reasonable size, filtering obscure characters is mandatory.

# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.

NOTE: sadly I did not go too much into model initialization in the video lecture, but it is quite important for good performance. The current code will train and work fine, but its convergence is slower because it starts off in a not great spot in the weight space. Please see [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for `# init all weights` comment, and especially how it calls the `_init_weights` function. Even more sadly, the code in this repo is a bit different in how it names and stores the various modules, so it's not possible to directly copy paste this code here. My current plan is to publish a supplementary video lecture and cover these parts, then I will also push the exact code changes to this repo. For now I'm keeping it as is so it is almost exactly what we actually covered in the video.

### License

MIT
