import torch
from nanochat.checkpoint_manager import load_model_from_dir
from reward_model import RewardModel

device = torch.device("cuda")

checkpoints_dir = "/workspace/nanochat_checkpoints_d20/sft_d20_v2_best"
REWARD_HEAD_PATH = "/workspace/nanochat_checkpoints_d20/reward_model.pt"

policy_model, tokenizer, meta_data = load_model_from_dir(checkpoints_dir, device, phase="train", model_tag="d20", step=36000)
reference_model, _, _ = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag="d20", step=36000)

for param in reference_model.parameters():
    param.requires_grad = False

reward_base_model, _, _ = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag="d20", step=36000)
reward_model = RewardModel(reward_base_model).to(device)
reward_model.reward_head.load_state_dict(torch.load(REWARD_HEAD_PATH))
for param in reward_model.parameters():
    param.requires_grad = False

value_base_model, _, _ = load_model_from_dir(checkpoints_dir, device, phase="train", model_tag="d20", step=36000)
value_model = RewardModel(value_base_model).to(device)

print("Policy model loaded, trainable")
print("Reference model loaded, frozen")
print("Reward model loaded, frozen, with trained weights")
print("Value model loaded, trainable (untrained, starts fresh)")

print(torch.cuda.memory_summary(device=device, abbreviated=True))

from datasets import load_dataset

train_ds = load_dataset('CarperAI/openai_summarize_comparisons', split='train')
example = train_ds[0]

prompt_text = f"Summarize this: {example['prompt']}\n\nSummary:"
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

generated_summary = tokenizer.decode(generated_tokens[len(prompt_ids):])
print("Generated summary:", generated_summary)

def score_with_reward_model(post, summary, reward_model, tokenizer, device):
    text = f"Summarize this: {post}\n\nSummary: {summary}"
    ids = tokenizer.encode(text)
    ids = ids[:2048]
    ids_tensor = torch.tensor([ids], device=device)
    return reward_model(ids_tensor)

reward_score = score_with_reward_model(example['prompt'], generated_summary, reward_model, tokenizer, device)
print("Reward score for generated summary:", reward_score.item())

def get_log_probs(model, ids):
    ids_tensor = torch.tensor([ids], device=device)
    inputs = ids_tensor[:, :-1]
    targets = ids_tensor[:, 1:]
    logits = model(inputs)
    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
    return target_log_probs

policy_model.eval()
with torch.no_grad():
    policy_log_probs = get_log_probs(policy_model, generated_tokens)
    reference_log_probs = get_log_probs(reference_model, generated_tokens)
policy_model.train()

kl_divergence = (policy_log_probs - reference_log_probs).mean()
print("KL divergence (policy vs reference):", kl_divergence.item())

value_prediction = value_model(torch.tensor([generated_tokens], device=device))
print("Value model's prediction:", value_prediction.item())

advantage = reward_score - value_prediction
print("Advantage (actual reward - predicted value):", advantage.item())
