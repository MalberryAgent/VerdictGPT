import torch
from nanochat.checkpoint_manager import load_model_from_dir, build_model
from datasets import load_dataset

device = torch.device("cuda")

def generate(model, tokenizer, prompt_ids):
    generated_tokens = list(prompt_ids)
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    with torch.no_grad():
        for next_token in model.generate(prompt_ids, max_tokens=60, top_k=50):
            if next_token == assistant_end:
                break
            generated_tokens.append(next_token)
    return generated_tokens

val_ds = load_dataset('CarperAI/openai_summarize_comparisons', split='valid1')
example = val_ds[3]
post = example['prompt']

base_model, tokenizer, _ = load_model_from_dir("/workspace/nanochat_checkpoints_d20/base_checkpoints", device, phase="eval", model_tag="d20", step=3320)
prompt_ids = tokenizer.encode(f"Summarize this: {post}\n\nSummary:")
base_out = tokenizer.decode(generate(base_model, tokenizer, prompt_ids)[len(prompt_ids):])
del base_model
torch.cuda.empty_cache()

sft_model, _, _ = load_model_from_dir("/workspace/nanochat_checkpoints_d20/sft_d20_v2_best", device, phase="eval", model_tag="d20", step=36000)
sft_out = tokenizer.decode(generate(sft_model, tokenizer, prompt_ids)[len(prompt_ids):])
del sft_model
torch.cuda.empty_cache()

ppo_model, _, _ = build_model("/workspace/nanochat_checkpoints_d20/ppo_checkpoints/d20", 10, device, phase="eval")
ppo_out = tokenizer.decode(generate(ppo_model, tokenizer, prompt_ids)[len(prompt_ids):])

print("=== POST ===")
print(post[:300])
print()
print("=== BASE MODEL ===")
print(base_out)
print()
print("=== SFT MODEL ===")
print(sft_out)
print()
print("=== PPO MODEL ===")
print(ppo_out)
