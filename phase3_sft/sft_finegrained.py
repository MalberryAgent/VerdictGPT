import os
import random
import torch
from nanochat.checkpoint_manager import load_model, save_checkpoint
from datasets import load_dataset
from sft_data import format_example

device = torch.device("cuda")

train_ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', split='train')
val_ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', split='validation')

BEST_DIR = "/workspace/nanochat_checkpoints/sft_finegrained_best/d12"

MAX_STEPS = 3000
GRAD_ACCUM = 8
VAL_EVERY = 250
VAL_BATCHES = 20

model, tokenizer, meta_data = load_model("base", device, phase="train")
optimizer = model.setup_optimizer(matrix_lr=0.002, embedding_lr=0.02, unembedding_lr=0.0004, scalar_lr=0.05)

def compute_loss(example, model, tokenizer, device):
    ids, mask = format_example(example, tokenizer)
    ids_tensor = torch.tensor([ids], device=device)
    mask_tensor = torch.tensor([mask], device=device)
    inputs = ids_tensor[:, :-1]
    targets = ids_tensor[:, 1:]
    target_mask = mask_tensor[:, 1:]
    masked_targets = targets.clone()
    masked_targets[target_mask == 0] = -1
    return model(inputs, targets=masked_targets)

def delete_checkpoint_files(checkpoint_dir, step):
    for prefix in ["model_", "meta_"]:
        for fname in os.listdir(checkpoint_dir):
            if fname.startswith(f"{prefix}{step:06d}"):
                os.remove(os.path.join(checkpoint_dir, fname))

shuffled_indices = list(range(len(train_ds)))
random.shuffle(shuffled_indices)

running_loss = 0.0
running_count = 0
best_val_loss = float("inf")
best_step = None

model.train()
for step in range(MAX_STEPS):
    example = train_ds[shuffled_indices[step]]
    loss = compute_loss(example, model, tokenizer, device)
    (loss / GRAD_ACCUM).backward()
    running_loss += loss.item()
    running_count += 1

    if (step + 1) % GRAD_ACCUM == 0:
        optimizer.step()
        optimizer.zero_grad()

    if (step + 1) % VAL_EVERY == 0:
        avg_train_loss = running_loss / running_count
        running_loss = 0.0
        running_count = 0

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for i in range(VAL_BATCHES):
                val_example = val_ds[i]
                val_loss = compute_loss(val_example, model, tokenizer, device)
                val_loss_total += val_loss.item()
        avg_val_loss = val_loss_total / VAL_BATCHES
        model.train()

        current_step = step + 1
        print(f"Step {current_step}/{MAX_STEPS} -- train loss: {avg_train_loss:.4f} -- val loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            if best_step is not None:
                delete_checkpoint_files(BEST_DIR, best_step)
            best_val_loss = avg_val_loss
            best_step = current_step
            best_meta = {"step": current_step, "val_loss": avg_val_loss, "model_config": meta_data["model_config"]}
            save_checkpoint(BEST_DIR, current_step, model.state_dict(), None, best_meta)
            print(f"  -- new best val loss, checkpoint saved at step {current_step}")

print(f"Scan complete. Best step: {best_step}, best val loss: {best_val_loss:.4f}")
