import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt import BigramLanguageModel, decode, model, device, encode

nsamples = 256
batch_size = 1
model_path = "./models/model-last.pt"

# load model from file
model.load_state_dict(torch.load(model_path))

print("model loaded")

# generate from the model
#context = encode("The industries success is based on")
context = encode("A temperature of 100°C (Celsius) is roughly equivalent to ")

context = torch.tensor(context, device=device, dtype=torch.float16).unsqueeze(0).repeat(batch_size, 1)
generated = model.generate(context, max_new_tokens=nsamples)
for result in generated:
    response = decode(result.tolist())
    print(response, end="\n\n")