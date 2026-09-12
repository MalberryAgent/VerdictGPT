import torch
from nanochat.checkpoint_manager import load_model_from_dir
from datasets import load_dataset

device = torch.device("cuda")

checkpoints_dir = "/workspace/nanochat_checkpoints/sft_checkpoints"
model, tokenizer, meta_data = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag="d12", step=1200)

val_ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', split='validation')
example = val_ds[3]

conversation = {
    "messages": [
        {"role": "user", "content": f"Summarize this: {example['post']}"},
        {"role": "assistant", "content": ""}
    ]
}
prompt_ids = tokenizer.render_for_completion(conversation)

generated_tokens = list(prompt_ids)
assistant_end = tokenizer.encode_special("<|assistant_end|>")
for next_token in model.generate(prompt_ids, max_tokens=60, top_k=50):
    if next_token == assistant_end:
        break
    generated_tokens.append(next_token)

model_summary = tokenizer.decode(generated_tokens[len(prompt_ids):])

print("=== Original post (first 300 chars) ===")
print(example['post'][:300])
print()
print("=== Human-written summary ===")
print(example['summary'])
print()
print("=== Model-generated summary ===")
print(model_summary)
