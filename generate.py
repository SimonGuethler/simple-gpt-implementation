import torch
import torch.nn as nn
from torch.nn import functional as F
from gpt import gpt 

nsamples = 128
batch_size = 4
model_path = './models/model_wiki/model-best-val.pt'


# initialize model
model = gpt.get_model()
# load checkpoint from file
model.load_state_dict(torch.load(model_path))

print("model loaded")

# generate from the model
context = gpt.encode("Animals can be categorized into vertebrates and ")

context = torch.tensor(context, device=gpt.device).unsqueeze(0).repeat(batch_size, 1)
generated = model.generate(context, max_new_tokens=nsamples, temperature=0.7)
for result in generated:
    response = gpt.decode(result.tolist())
    print(response, end="\n\n")