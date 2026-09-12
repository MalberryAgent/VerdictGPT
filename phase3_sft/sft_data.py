from datasets import load_dataset
from nanochat.tokenizer import get_tokenizer

def format_example(example, tokenizer):
    conversation = {
        "messages": [
            {"role": "user", "content": f"Summarize this: {example['post']}"},
            {"role": "assistant", "content": example['summary']}
        ]
    }
    ids, mask = tokenizer.render_conversation(conversation)
    return ids, mask

if __name__ == "__main__":
    ds = load_dataset('vwxyzjn/summarize_from_feedback_tldr_3_filtered', split='train')
    tokenizer = get_tokenizer()

    ids, mask = format_example(ds[5], tokenizer)
    print("Example 5 -- tokens:", len(ids), "| trained tokens:", sum(mask))
