import torch
import torch.nn.functional as F
from nanochat.checkpoint_manager import load_model_from_dir
from datasets import load_dataset
from reward_model import RewardModel

device = torch.device("cuda")

checkpoints_dir = "/workspace/nanochat_checkpoints_d20/sft_d20_v2_best"
REWARD_HEAD_PATH = "/workspace/nanochat_checkpoints_d20/reward_model.pt"

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

NUM_STEPS = 20

for step in range(NUM_STEPS):
    example = train_ds[step]
    generated_tokens, prompt_len = generate_summary(example['prompt'], policy_model, tokenizer, device)
    generated_summary = tokenizer.decode(generated_tokens[prompt_len:])

    with torch.no_grad():
        reward_score = score_with_reward_model(example['prompt'], generated_summary, reward_model, tokenizer, device)
        old_policy_log_probs = get_log_probs(policy_model, generated_tokens)
        reference_log_probs = get_log_probs(reference_model, generated_tokens)

    kl_penalty = (old_policy_log_probs - reference_log_probs).mean()
    adjusted_reward = reward_score - KL_COEFF * kl_penalty

    value_prediction = value_model(torch.tensor([generated_tokens], device=device))
    advantage = (adjusted_reward - value_prediction).detach()

    new_policy_log_probs = get_log_probs(policy_model, generated_tokens)
    ratio = torch.exp(new_policy_log_probs - old_policy_log_probs).mean()

    policy_loss = -(ratio * advantage)
    value_loss = F.mse_loss(value_prediction, adjusted_reward.detach())

    policy_loss.backward()
    policy_optimizer.step()
    policy_optimizer.zero_grad()

    value_loss.backward()
    value_optimizer.step()
    value_optimizer.zero_grad()

    print(f"Step {step+1}/{NUM_STEPS} -- reward: {reward_score.item():.1f} -- value_pred: {value_prediction.item():.1f} -- KL: {kl_penalty.item():.4f} -- policy_loss: {policy_loss.item():.1f} -- value_loss: {value_loss.item():.1f}")

print("Test run complete")
