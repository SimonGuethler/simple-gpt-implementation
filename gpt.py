import glob, time, pickle, os

import torch
import torch.nn as nn
from torch.nn import functional as F

from char_filter import char_filter

# config
data_path = './input'
data_cache_path = './data_cache.pkl'
# ------------

# model parameters
block_size = 256 # what is the maximum context length for predictions?
n_embd = 384 # embedding size (dimensionality of the hidden state)
n_head = 6 # number of heads in multi-head attention in pytorch
n_layer = 6 # number of layers in the transformer model
dropout = 0.2 # dropout rate (probability of zeroing out activations)
# ------------

torch.manual_seed(1337)

char_filter_list = None
def sanitize_text(text: str) -> str:
    """Removes all characters except for ones in char_filter_list, which gets set in read_data()"""
    text = ''.join([c for c in text if c in char_filter_list])
    text.replace("()", "").replace("[]", "").replace("{}", "").replace("<>", "").replace("  ", " ")
    return text

def read_data(path: str) -> dict:
    """Reads data from path, splits into train and val, and returns a dictionary with the following keys:
    train_data: tensor of integers representing the training data
    val_data: tensor of integers representing the validation data
    vocab_size: number of unique characters in the data
    stoi: dictionary mapping characters to integers
    itos: dictionary mapping integers to characters
    """
    global char_filter_list
    char_filter_list, texts = char_filter(path)
    train_data = ""
    val_data = ""
    for text in texts:
        text = sanitize_text(text)
        # split into train and val
        n = int(0.9*len(text)) # first 90% will be train, rest val
        train_data += text[:n]
        val_data += text[n:]
    
    text = train_data + val_data
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    # decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]
    return {"train_data": train_data, "val_data": val_data, "vocab_size": vocab_size, "stoi": stoi, "itos": itos}


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_head, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, dropout, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, n_embd, n_head, n_layer, vocab_size, dropout, block_size, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def top_k_logits(logits, k) -> torch.Tensor:
        if k == 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1]
        return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)
        
    def generate(self, idx, max_new_tokens, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] / temperature # becomes (B, C)
            # apply top-k sampling
            # logits = self.top_k_logits(logits, k=top_k)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class GPT:
    def __init__(self, data_path, data_cache_path, block_size, n_embd: int = None, n_head: int = None, n_layer: int = None, dropout: float = None):
        self.data_path = data_path
        self.data_cache_path = data_cache_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.block_size = block_size

        loadt1 = time.time()
        if os.path.isfile(self.data_cache_path):
            with open(self.data_cache_path, 'rb') as fp:
                data = pickle.load(fp)
                n_embd = data["n_embd"]
                n_head = data["n_head"]
                n_layer = data["n_layer"]
                dropout = data["n_dropout"]
                print("pickle", end=" ")
        else:
            if not all([n_embd, n_head, n_layer, dropout]):
                raise ValueError("Must specify all of n_embd, n_head, n_layer, dropout when training on new data")
            data = read_data(self.data_path)
            data["n_embd"] = n_embd
            data["n_head"] = n_head
            data["n_layer"] = n_layer
            data["n_dropout"] = dropout
            with open(self.data_cache_path, 'wb') as fp:
                pickle.dump(data, fp)
                print("txt", end=" ")
        
        loadt2 = time.time()
        print(f"data load time: {loadt2-loadt1:.2f} seconds")
        print(f"model params: {n_layer} layers, {n_head} heads, {n_embd} embedding size, {dropout} dropout")

        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.vocab_size = data["vocab_size"]
        stoi = data["stoi"]
        itos = data["itos"]
        

        self.encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        self.decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

        self.inputdata = { 'train': data["train_data"], 'val': data["val_data"] }
        print("train:", len(self.inputdata["train"]), "val:", len(self.inputdata["val"]), "vocab:", self.vocab_size)

        self.model = None

    # data loading
    def get_batch(self, set_name: str, batch_size: int):
        # generate a small batch of data of inputs x and targets y
        ix = torch.randint(len(self.inputdata[set_name]) - self.block_size, (batch_size,))
        x = torch.stack([self.inputdata[set_name][i:i+self.block_size] for i in ix])
        y = torch.stack([self.inputdata[set_name][i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self, eval_iters: int, batch_size: int):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split, batch_size=batch_size)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def get_model(self, gpu: bool = True) -> GPTLanguageModel:
        if self.model is None:
            self.model = GPTLanguageModel(
                n_embd=self.n_embd,
                n_head=self.n_head,
                n_layer=self.n_layer,
                vocab_size=self.vocab_size,
                dropout=self.dropout,
                block_size=self.block_size,
                device=self.device
            )
            # print the number of parameters in the model
            print('initialized gpt model with', sum(p.numel() for p in self.model.parameters())/1e6, 'M parameters')
            if gpu:
                self.model = self.model.to(self.device)
                #model = model.half().to(device) # for fp16
        return self.model

gpt = GPT(
    data_path=data_path,
    data_cache_path=data_cache_path,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
)