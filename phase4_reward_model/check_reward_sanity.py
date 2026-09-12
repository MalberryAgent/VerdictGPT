import torch
from nanochat.checkpoint_manager import load_model_from_dir
from reward_model import RewardModel

device = torch.device("cuda")
checkpoints_dir = "/workspace/nanochat_checkpoints_d20/sft_d20_v2_best"
base_model, tokenizer, meta_data = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag="d20", step=36000)
reward_model = RewardModel(base_model).to(device)
reward_model.reward_head.load_state_dict(torch.load("/workspace/nanochat_checkpoints_d20/reward_model.pt"))

def score(text):
    ids = tokenizer.encode(text)[:2048]
    return reward_model(torch.tensor([ids], device=device)).item()

post = "A test post about a minor disagreement with a coworker."
normal_summary = "I had a small disagreement with my coworker about scheduling."
garbage_summary = "fuck " * 40

print("Normal summary score:", score(f"Summarize this: {post}\n\nSummary: {normal_summary}"))
print("Garbage summary score:", score(f"Summarize this: {post}\n\nSummary: {garbage_summary}"))
