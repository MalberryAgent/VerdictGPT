import torch
from nanochat.checkpoint_manager import build_model
from datasets import load_dataset
from nanochat.tokenizer import get_tokenizer

device = torch.device("cuda")

checkpoint_dir = "/workspace/nanochat_checkpoints_d20/ppo_checkpoints/d20"
model, tokenizer, meta_data = build_model(checkpoint_dir, 10, device, phase="eval")

val_ds = load_dataset('CarperAI/openai_summarize_comparisons', split='valid1')

for idx in [0, 5, 10]:
    example = val_ds[idx]
    prompt_text = f"Summarize this: {example['prompt']}\n\nSummary:"
    prompt_ids = tokenizer.encode(prompt_text)
    generated_tokens = list(prompt_ids)
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    with torch.no_grad():
        for next_token in model.generate(prompt_ids, max_tokens=60, top_k=50):
            if next_token == assistant_end:
                break
            generated_tokens.append(next_token)
    summary = tokenizer.decode(generated_tokens[len(prompt_ids):])
    print(f"=== Example {idx} ===")
    print("Generated:", summary)
    print()
