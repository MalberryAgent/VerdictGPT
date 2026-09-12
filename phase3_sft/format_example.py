from datasets import load_dataset
from nanochat.tokenizer import get_tokenizer

ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', split='train')
example = ds[0]

conversation = {
    "messages": [
        {"role": "user", "content": f"Summarize this: {example['post']}"},
        {"role": "assistant", "content": example['summary']}
    ]
}

tokenizer = get_tokenizer()
ids, mask = tokenizer.render_conversation(conversation)

print("Number of tokens:", len(ids))
print("Number of tokens being trained on (mask=1):", sum(mask))
print()
print(tokenizer.visualize_tokenization(ids, mask))
