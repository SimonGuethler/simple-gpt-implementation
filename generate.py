import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt import decode, model, device, encode, models_path

nsamples = 128
batch_size = 1
model_path = models_path + "/model-last.pt"
# load model from file
model.load_state_dict(torch.load(model_path))

print("model loaded")

# generate from the model
context = encode("Animals can be categorized into vertebrates and ")

context = torch.tensor(context, device=device).unsqueeze(0).repeat(batch_size, 1)
generated = model.generate(context, max_new_tokens=nsamples)
for result in generated:
    response = decode(result.tolist())
    print(response, end="\n\n")