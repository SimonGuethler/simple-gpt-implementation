import os, time
import torch
from gpt import estimate_loss, eval_interval, max_iters, learning_rate, model, model, get_batch, inputdata, models_path
from optimizer import Adam16

optimizer = Adam16(model.parameters(), lr=learning_rate)
# scaler = torch.cuda.amp.GradScaler()


if not os.path.isdir(models_path):
    os.makedirs(models_path) 

last_losses = None
for iter in range(max_iters):
    #with torch.autocast(device_type='cuda', dtype=torch.float16):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:

        t1 = time.time()
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        losses = estimate_loss()
        t2 = time.time()

        if last_losses is None:
            last_losses = losses
        
        print(f"step {iter:04f}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if losses["train"] < last_losses["train"]:
            torch.save(model.state_dict(), os.path.join(models_path, "model-best-train.pt"))
        if losses["val"] < last_losses["val"]:
            torch.save(model.state_dict(), os.path.join(models_path, "model-best-val.pt"))
        torch.save(model.state_dict(), os.path.join(models_path, "model-last.pt"))
        t3 = time.time()
        print(f"evaluation took {t2-t1:.3f} seconds. model saved in {t3-t2:.3f} seconds.")

    # sample a batch of data
    xb, yb = get_batch(inputdata['train'])

    # evaluate the loss

    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    logits, loss = model(xb, yb) 
    
    # logits, loss = m(xb, yb)
    
    # logits, loss = model(xb, yb)
    
    # train

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()
    # optimizer.zero_grad() # set_to_none=True here can modestly improve performance