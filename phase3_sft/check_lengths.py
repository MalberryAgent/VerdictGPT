from datasets import load_dataset
from nanochat.tokenizer import get_tokenizer

ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', split='train')
tokenizer = get_tokenizer()

lengths = []
for i in range(500):
    example = ds[i]
    conversation = {
        "messages": [
            {"role": "user", "content": f"Summarize this: {example['post']}"},
            {"role": "assistant", "content": example['summary']}
        ]
    }
    ids, mask = tokenizer.render_conversation(conversation, max_tokens=100000)
    lengths.append(len(ids))

print("Checked 500 examples")
print("Shortest:", min(lengths))
print("Longest:", max(lengths))
print("Average:", sum(lengths) / len(lengths))
print("Number over 2048:", sum(1 for l in lengths if l > 2048))
