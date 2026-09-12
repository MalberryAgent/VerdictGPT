import torch
import torch.nn.functional as F
from nanochat.checkpoint_manager import load_model_from_dir
from datasets import load_dataset
from reward_model import RewardModel

device = torch.device("cuda")

checkpoints_dir = "/workspace/nanochat_checkpoints_d20/sft_d20_v2_best"
base_model, tokenizer, meta_data = load_model_from_dir(checkpoints_dir, device, phase="train", model_tag="d20", step=36000)

reward_model = RewardModel(base_model).to(device)
optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-4)

train_ds = load_dataset('CarperAI/openai_summarize_comparisons', split='train')
val_ds = load_dataset('CarperAI/openai_summarize_comparisons', split='valid1')

def format_and_score(post, summary, reward_model, tokenizer, device):
    text = f"Summarize this: {post}\n\nSummary: {summary}"
    ids = tokenizer.encode(text)
    ids = ids[:2048]
    ids_tensor = torch.tensor([ids], device=device)
    return reward_model(ids_tensor)

GRAD_ACCUM = 16
NUM_STEPS = 30000
VAL_EVERY = 3000
VAL_BATCHES = 200
SAVE_PATH = "/workspace/nanochat_checkpoints_d20/reward_model.pt"

running_loss = 0.0
running_correct = 0
running_count = 0

reward_model.train()
for step in range(NUM_STEPS):
    example = train_ds[step]
    chosen_score = format_and_score(example['prompt'], example['chosen'], reward_model, tokenizer, device)
    rejected_score = format_and_score(example['prompt'], example['rejected'], reward_model, tokenizer, device)

    loss = -F.logsigmoid(chosen_score - rejected_score)
    (loss / GRAD_ACCUM).backward()

    if (step + 1) % GRAD_ACCUM == 0:
        optimizer.step()
        optimizer.zero_grad()

    running_loss += loss.item()
    running_correct += (chosen_score > rejected_score).item()
    running_count += 1

    if (step + 1) % VAL_EVERY == 0:
        avg_loss = running_loss / running_count
        accuracy = running_correct / running_count
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        reward_model.eval()
        val_correct = 0
        with torch.no_grad():
            for i in range(VAL_BATCHES):
                val_example = val_ds[i]
                cs = format_and_score(val_example['prompt'], val_example['chosen'], reward_model, tokenizer, device)
                rs = format_and_score(val_example['prompt'], val_example['rejected'], reward_model, tokenizer, device)
                val_correct += (cs > rs).item()
        val_accuracy = val_correct / VAL_BATCHES
        reward_model.train()

        print(f"Step {step + 1}/{NUM_STEPS} -- train loss: {avg_loss:.4f} -- train acc: {accuracy:.2f} -- val acc: {val_accuracy:.2f}")

        torch.save(reward_model.reward_head.state_dict(), SAVE_PATH)
        print(f"  -- reward head saved to {SAVE_PATH}")

print("Training run complete")
