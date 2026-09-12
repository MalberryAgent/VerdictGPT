import torch
from nanochat.checkpoint_manager import load_model

device = torch.device("cuda")
model, tokenizer, meta_data = load_model("base", device, phase="eval")

print("Model loaded successfully")

prompt = "The best way to learn a new skill is"
tokens = tokenizer.encode(prompt)

generated_tokens = list(tokens)
for next_token in model.generate(tokens, max_tokens=30, top_k=50, temperature=1.5):
    generated_tokens.append(next_token)

output_text = tokenizer.decode(generated_tokens)
print("Generated text:", output_text)
