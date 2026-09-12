import os
import random
import torch
from nanochat.checkpoint_manager import load_model, save_checkpoint, load_checkpoint, build_model, find_last_step
from datasets import load_dataset
from sft_data import format_example

device = torch.device("cuda")

train_ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', split='train')
val_ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', split='validation')

RESUME_DIR = "/workspace/nanochat_checkpoints_d20/sft_d20_v2_resume/d20"
BEST_DIR = "/workspace/nanochat_checkpoints_d20/sft_d20_v2_best/d20"

MAX_STEPS = 150000
GRAD_ACCUM = 128
VAL_EVERY = 2000
VAL_BATCHES = 20
PATIENCE = 10
KEEP_BEST_CHECKPOINTS = 2

start_step = 0
best_val_loss = float("inf")
patience_counter = 0

try:
    resume_step = find_last_step(RESUME_DIR)
    print(f"Found resume checkpoint at step {resume_step}, resuming")
    model, tokenizer, meta_data = build_model(RESUME_DIR, resume_step, device, phase="train")
    optimizer = model.setup_optimizer(matrix_lr=0.0079, embedding_lr=0.079, unembedding_lr=0.0016, scalar_lr=0.198)
    _, optimizer_data, _ = load_checkpoint(RESUME_DIR, resume_step, device, load_optimizer=True)
    optimizer.load_state_dict(optimizer_data)
    start_step = meta_data["train_step"]
    best_val_loss = meta_data["best_val_loss"]
    patience_counter = meta_data["patience_counter"]
    print(f"Resumed at train_step={start_step}, best_val_loss={best_val_loss:.4f}, patience_counter={patience_counter}")
except FileNotFoundError:
    print("No resume checkpoint found, starting fresh from base model")
    model, tokenizer, meta_data = load_model("base", device, phase="train")
    optimizer = model.setup_optimizer(matrix_lr=0.0079, embedding_lr=0.079, unembedding_lr=0.0016, scalar_lr=0.198)

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
    for prefix in ["model_", "meta_", "optim_"]:
        for fname in os.listdir(checkpoint_dir):
            if fname.startswith(f"{prefix}{step:06d}"):
                os.remove(os.path.join(checkpoint_dir, fname))

def cleanup_old_best_checkpoints(checkpoint_dir, keep_steps):
    all_steps = set()
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("model_"):
            step_num = int(fname.split("_")[1].split(".")[0])
            all_steps.add(step_num)
    steps_to_delete = sorted(all_steps)[:-keep_steps] if len(all_steps) > keep_steps else []
    for step_num in steps_to_delete:
        delete_checkpoint_files(checkpoint_dir, step_num)

shuffled_indices = list(range(len(train_ds)))
random.shuffle(shuffled_indices)

running_loss = 0.0
running_count = 0
previous_resume_step = start_step if start_step > 0 else None

model.train()
for step in range(start_step, MAX_STEPS):
    pos_in_epoch = step % len(train_ds)
    if pos_in_epoch == 0 and step != start_step:
        random.shuffle(shuffled_indices)
        print(f"  -- reshuffled dataset at step {step}")

    example = train_ds[shuffled_indices[pos_in_epoch]]
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

        is_new_best = avg_val_loss < best_val_loss
        if is_new_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        resume_meta = {
            "train_step": current_step,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "model_config": meta_data["model_config"],
        }
        save_checkpoint(RESUME_DIR, current_step, model.state_dict(), optimizer.state_dict(), resume_meta)
        if previous_resume_step is not None:
            delete_checkpoint_files(RESUME_DIR, previous_resume_step)
        previous_resume_step = current_step

        if is_new_best:
            best_meta = {"step": current_step, "val_loss": avg_val_loss, "model_config": meta_data["model_config"]}
            save_checkpoint(BEST_DIR, current_step, model.state_dict(), None, best_meta)
            print(f"  -- new best val loss, checkpoint saved at step {current_step}")
            cleanup_old_best_checkpoints(BEST_DIR, KEEP_BEST_CHECKPOINTS)
        else:
            print(f"  -- no improvement ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"Early stopping: no improvement for {PATIENCE} checks in a row")
            break

print("Training run complete")
