import torch
from nanochat.checkpoint_manager import load_model_from_dir
from datasets import load_dataset

device = torch.device("cuda")

checkpoints_dir = "/workspace/nanochat_checkpoints_d20/sft_d20_best"
model, tokenizer, meta_data = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag="d20", step=6000)

val_ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', split='validation')

for idx in [3, 10, 25]:
    example = val_ds[idx]
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

    print(f"=== Example {idx} ===")
    print("Human summary:", example['summary'])
    print("Model summary:", model_summary)
    print()
