import torch
from nanochat.checkpoint_manager import load_model, save_checkpoint
from datasets import load_dataset
from sft_data import format_example

device = torch.device("cuda")
model, tokenizer, meta_data = load_model("base", device, phase="train")
optimizer = model.setup_optimizer(matrix_lr=0.002, embedding_lr=0.02, unembedding_lr=0.0004, scalar_lr=0.05)

train_ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', split='train')
val_ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', split='validation')

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

NUM_STEPS = 2000
GRAD_ACCUM = 8
VAL_EVERY = 200
VAL_BATCHES = 20
CHECKPOINT_DIR = "/workspace/nanochat_checkpoints/sft_checkpoints/d12"

running_loss = 0.0
best_val_loss = float("inf")

model.train()
for step in range(NUM_STEPS):
    example = train_ds[step % len(train_ds)]
    loss = compute_loss(example, model, tokenizer, device)
    (loss / GRAD_ACCUM).backward()
    running_loss += loss.item()

    if (step + 1) % GRAD_ACCUM == 0:
        optimizer.step()
        optimizer.zero_grad()

    if (step + 1) % VAL_EVERY == 0:
        avg_train_loss = running_loss / VAL_EVERY
        running_loss = 0.0

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for i in range(VAL_BATCHES):
                val_example = val_ds[i]
                val_loss = compute_loss(val_example, model, tokenizer, device)
                val_loss_total += val_loss.item()
        avg_val_loss = val_loss_total / VAL_BATCHES
        model.train()

        print(f"Step {step + 1}/{NUM_STEPS} -- train loss: {avg_train_loss:.4f} -- val loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_data = model.state_dict()
            save_meta = {"step": step + 1, "val_loss": avg_val_loss, "model_config": meta_data["model_config"]}
            save_checkpoint(CHECKPOINT_DIR, step + 1, model_data, None, save_meta)
            print(f"  -- new best val loss, checkpoint saved at step {step + 1}")

print("Training run complete")
