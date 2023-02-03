import os, time, datetime, sys, json, signal
import torch
from gpt import gpt
from optimizer import Adam16

average_power_usage = 550 # watts

models_path = './models/model'
# hyperparameters for training (will be written to the data cache file)
batch_size = 16 # how many independent sequences will we process in parallel?
eval_interval = 2000 # how often to evaluate the model on train and val sets
max_iters = 200_000
learning_rate = 3e-4 # 3e-4 is the default in the original paper 
eval_iters = 200 # how many batches to use for evaluation
# --------------------------------


model = gpt.get_model()
model.train()

# optimizer = Adam16(model.parameters(), lr=learning_rate) # for fp16
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# scaler = torch.cuda.amp.GradScaler()

if not os.path.isdir(models_path):
    os.makedirs(models_path)


t0 = time.time()
best_score = None

iter_range = range(max_iters)

# resume training
if "continue" in sys.argv:
    model.load_state_dict(torch.load(os.path.join(models_path, "model-last.pt")))
    with open(os.path.join(models_path, "training-state.json"), "r", encoding="utf-8") as f:
        training_state = json.load(f)
    iter_range = range(training_state["iter"], max_iters)
    t0 = time.time() - training_state["time"]
    best_score = training_state["best_score"]

iter = 0
t2 = 0
# attach ctrl+c handler
def signal_handler(sig, frame):
    torch.save(model.state_dict(), os.path.join(models_path, "model-last.pt"))

    with(open(os.path.join(models_path, "training-state.json"), "w", encoding="utf-8")) as f:
        json.dump({"iter": iter, "time": int(t2-t0), "best_score": best_score}, f)
    
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


for iter in iter_range:
    #with torch.autocast(device_type='cuda', dtype=torch.float16): # for fp16
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:

        t1 = time.time()
        # with torch.autocast(device_type='cuda', dtype=torch.float16): # for fp16?
        score = gpt.estimate_loss(eval_iters=eval_iters, batch_size=batch_size)
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

        with(open(os.path.join(models_path, "training-state.json"), "w", encoding="utf-8")) as f:
            json.dump({"iter": iter, "time": int(t2-t0), "best_score": best_score}, f)
        
        t3 = time.time()
        
        if iter > 0:
            remaining_time = ((time.time()-t0)/60/60) / iter * (max_iters-iter) # h
        else:
            remaining_time = 0.0
        
        power_used = (time.time()-t0)/60/60*average_power_usage/1000 # kWh
        training_time = str(datetime.timedelta(seconds=int(time.time()-t0)))
        
        print(f" evaluation took {t2-t1:.2f} seconds. model saved in {t3-t2:.2f} seconds. Total time wasted training: {training_time}, approx. {power_used:.3f} kWh used, remaining time: {remaining_time:.2f} hours.")

    # sample a batch of data
    xb, yb = gpt.get_batch("train", batch_size=batch_size)

    # evaluate the loss

    # with torch.autocast(device_type='cuda', dtype=torch.float16): # for fp16?
    logits, loss = model(xb, yb) 
    # up until here with first torch autocast, test if it works only at eval time?
    
    # train
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # scaler.scale(loss).backward()
    # scaler.step(optimizer)
    # scaler.update()
    # optimizer.zero_grad() # set_to_none=True here can modestly improve performance
