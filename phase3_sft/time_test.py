import time
import torch
from nanochat.checkpoint_manager import load_model
from datasets import load_dataset
from sft_data import format_example

device = torch.device("cuda")
model, tokenizer, meta_data = load_model("base", device, phase="train")
optimizer = model.setup_optimizer(matrix_lr=0.002, embedding_lr=0.02, unembedding_lr=0.0004, scalar_lr=0.05)

train_ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', split='train')

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

model.train()
torch.cuda.synchronize()
start = time.time()

TEST_STEPS = 200
for step in range(TEST_STEPS):
    example = train_ds[step]
    loss = compute_loss(example, model, tokenizer, device)
    (loss / 8).backward()
    if (step + 1) % 8 == 0:
        optimizer.step()
        optimizer.zero_grad()

torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Time for {TEST_STEPS} steps: {elapsed:.1f} seconds")
print(f"Time per step: {elapsed/TEST_STEPS:.3f} seconds")

full_epoch_steps = len(train_ds)
est_epoch_seconds = (elapsed/TEST_STEPS) * full_epoch_steps
print(f"Estimated time for 1 full epoch ({full_epoch_steps} examples): {est_epoch_seconds/3600:.2f} hours")
