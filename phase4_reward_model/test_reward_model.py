import torch
from nanochat.checkpoint_manager import load_model_from_dir
from reward_model import RewardModel

device = torch.device("cuda")

checkpoints_dir = "/workspace/nanochat_checkpoints_d20/sft_d20_v2_best"
base_model, tokenizer, meta_data = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag="d20", step=36000)

reward_model = RewardModel(base_model).to(device)

test_text = "Summarize this: some example post here"
ids = tokenizer.encode(test_text)
ids_tensor = torch.tensor([ids], device=device)

score = reward_model(ids_tensor)
print("Reward score:", score.item())
