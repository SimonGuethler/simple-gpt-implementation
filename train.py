import os, time
import torch
from gpt import estimate_loss, eval_interval, max_iters, learning_rate, model, model, get_batch, inputdata, models_path
from optimizer import Adam16

# optimizer = Adam16(model.parameters(), lr=learning_rate) # for fp16
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# scaler = torch.cuda.amp.GradScaler()


if not os.path.isdir(models_path):
    os.makedirs(models_path) 

t0 = time.time()
best_score = None
for iter in range(max_iters):
    #with torch.autocast(device_type='cuda', dtype=torch.float16): # for fp16
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:

        t1 = time.time()
        # with torch.autocast(device_type='cuda', dtype=torch.float16): # for fp16?
        score = estimate_loss()
        t2 = time.time()

        if best_score is None:
            best_score = score
        
        print(f"step {iter}: train loss {score['train']:.4f}, val loss {score['val']:.4f}")
        
        if score["train"] < best_score["train"]:
            torch.save(model.state_dict(), os.path.join(models_path, "model-best-train.pt"))
            best_score["train"] = score["train"]
        
        if score["val"] < best_score["val"]:
            torch.save(model.state_dict(), os.path.join(models_path, "model-best-val.pt"))
            best_score["val"] = score["val"]
        
        torch.save(model.state_dict(), os.path.join(models_path, "model-last.pt"))
        t3 = time.time()
        print(f"evaluation took {t2-t1:.2f} seconds. model saved in {t3-t2:.2f} seconds. Total time wasted training: {(time.time()-t0)/60:.2f} minutes. ({(time.time()-t0)/60/60*650/1000:.3f} kWh used)")

    # sample a batch of data
    xb, yb = get_batch(inputdata['train'])

    # evaluate the loss

    # with torch.autocast(device_type='cuda', dtype=torch.float16): # for fp16?
    logits, loss = model(xb, yb) 
    # up until here with first torch autocast, test if it works only at eval time?
    
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