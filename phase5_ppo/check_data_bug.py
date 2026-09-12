from datasets import load_dataset

train_ds = load_dataset('CarperAI/openai_summarize_comparisons', split='train')

for i in range(8):
    example = train_ds[i]
    print(f"[{i}] {example['prompt'][:60]}")
