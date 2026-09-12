import torch
import torch.nn.functional as F
from nanochat.checkpoint_manager import load_model_from_dir, save_checkpoint
from datasets import load_dataset
from reward_model import RewardModel

device = torch.device("cuda")

checkpoints_dir = "/workspace/nanochat_checkpoints_d20/sft_d20_v2_best"
REWARD_HEAD_PATH = "/workspace/nanochat_checkpoints_d20/reward_model.pt"
SAVE_DIR = "/workspace/nanochat_checkpoints_d20/ppo_checkpoints/d20"

policy_model, tokenizer, meta_data = load_model_from_dir(checkpoints_dir, device, phase="train", model_tag="d20", step=36000)
reference_model, _, _ = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag="d20", step=36000)
for param in reference_model.parameters():
    param.requires_grad = False

reward_model = RewardModel(reference_model).to(device)
reward_model.reward_head.load_state_dict(torch.load(REWARD_HEAD_PATH))
for param in reward_model.reward_head.parameters():
    param.requires_grad = False

value_model = RewardModel(policy_model, detach_input=True).to(device)

policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-6)
value_optimizer = torch.optim.AdamW(value_model.reward_head.parameters(), lr=1e-4)

train_ds = load_dataset('CarperAI/openai_summarize_comparisons', split='train')

KL_COEFF = 0.1
CLIP_RANGE = 0.2
ROLLOUT_SIZE = 8
INNER_EPOCHS = 4
OUTER_ROLLOUTS = 10
SAVE_EVERY = 10
MAX_GRAD_NORM = 1.0

def generate_summary(post, policy_model, tokenizer, device):
    prompt_text = f"Summarize this: {post}\n\nSummary:"
    prompt_ids = tokenizer.encode(prompt_text)
    generated_tokens = list(prompt_ids)
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    policy_model.eval()
    with torch.no_grad():
        for next_token in policy_model.generate(prompt_ids, max_tokens=60, top_k=50):
            if next_token == assistant_end:
                break
            generated_tokens.append(next_token)
    policy_model.train()
    return generated_tokens, len(prompt_ids)

def score_with_reward_model(post, summary, reward_model, tokenizer, device):
    text = f"Summarize this: {post}\n\nSummary: {summary}"
    ids = tokenizer.encode(text)[:2048]
    ids_tensor = torch.tensor([ids], device=device)
    return reward_model(ids_tensor)

def get_log_probs(model, ids):
    ids_tensor = torch.tensor([ids], device=device)
    inputs = ids_tensor[:, :-1]
    targets = ids_tensor[:, 1:]
    logits = model(inputs)
    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
    return target_log_probs

import random
shuffled_indices = list(range(len(train_ds)))
random.shuffle(shuffled_indices)
example_idx = 0

for rollout in range(OUTER_ROLLOUTS):
    batch = []
    for i in range(ROLLOUT_SIZE):
        example = train_ds[shuffled_indices[example_idx % len(shuffled_indices)]]
        example_idx += 1

        generated_tokens, prompt_len = generate_summary(example['prompt'], policy_model, tokenizer, device)
        generated_summary = tokenizer.decode(generated_tokens[prompt_len:])
        print(f"    [{i}] {generated_summary[:80]}")

        with torch.no_grad():
            reward_score = score_with_reward_model(example['prompt'], generated_summary, reward_model, tokenizer, device)
            old_log_probs = get_log_probs(policy_model, generated_tokens).mean()
            ref_log_probs = get_log_probs(reference_model, generated_tokens).mean()
            kl_penalty = old_log_probs - ref_log_probs
            adjusted_reward = reward_score - KL_COEFF * kl_penalty
            value_pred = value_model(torch.tensor([generated_tokens], device=device))
            advantage = adjusted_reward - value_pred

        batch.append({
            "tokens": generated_tokens,
            "old_log_probs": old_log_probs,
            "advantage": advantage,
            "target": adjusted_reward,
        })

    avg_reward = sum(b["target"].item() for b in batch) / len(batch)

    advantages = torch.stack([b["advantage"] for b in batch])
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    for b in batch:
        b["advantage"] = (b["advantage"] - adv_mean) / adv_std

    for inner_epoch in range(INNER_EPOCHS):
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()

        for b in batch:
            new_log_probs = get_log_probs(policy_model, b["tokens"]).mean()
            new_value_pred = value_model(torch.tensor([b["tokens"]], device=device))

            ratio = torch.exp(new_log_probs - b["old_log_probs"])
            unclipped = ratio * b["advantage"]
            clipped = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * b["advantage"]
            policy_loss = -torch.min(unclipped, clipped) / ROLLOUT_SIZE
            value_loss = F.mse_loss(new_value_pred, b["target"]) / ROLLOUT_SIZE

            policy_loss.backward()
            value_loss.backward()

        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(value_model.reward_head.parameters(), MAX_GRAD_NORM)

        policy_optimizer.step()
        value_optimizer.step()

    print(f"Rollout {rollout + 1}/{OUTER_ROLLOUTS} -- avg reward: {avg_reward:.1f} -- adv_std: {adv_std.item():.1f}")

    if (rollout + 1) % SAVE_EVERY == 0:
        save_checkpoint(SAVE_DIR, rollout + 1, policy_model.state_dict(), None, {"rollout": rollout + 1, "model_config": meta_data["model_config"]})
        print(f"  -- checkpoint saved at rollout {rollout + 1}")

print("Training run complete")
