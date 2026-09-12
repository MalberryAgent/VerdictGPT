from datasets import load_dataset

ds = load_dataset('CarperAI/openai_summarize_comparisons', split='train')
print("Train size:", len(ds))

val_ds = load_dataset('CarperAI/openai_summarize_comparisons', split='valid1')
print("Valid1 size:", len(val_ds))
