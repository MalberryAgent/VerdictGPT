# VerdictGPT — Full Phase Specifications

## Phase 0 — Setup ✅ DONE
RunPod account, GitHub repo, W&B account, persistent volume, environment verified (GPU visibility, CUDA, package versions, volume writability, repo state, W&B auth, end-to-end round trip).

## Phase 1 — Foundations ✅ DONE
- Karpathy, "Let's build GPT" — https://www.youtube.com/watch?v=kCc8FmEb1nY — watched, notes taken, passed a 6-question conceptual quiz (attention/Q-K-V, scaling by sqrt(d_k), causal masking, positional encoding, gradients).
- nanochat structure skimmed and mapped to file responsibilities.
- Went through `nanochat/gpt.py`, `tokenizer.py`, `scripts/base_train.py`, `nanochat/optim.py` line-by-line with Claude Code in tutor mode.
- Reference (dip in as needed): Karpathy "Neural Networks: Zero to Hero" — https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

**Exit criterion met:** can explain attention mechanics, causal masking, positional encoding, and what a gradient does, in my own words, without checking notes.

## Phase 2 — Base model ✅ DONE
- Trained via nanochat's `speedrun_4090.sh` (modified copy of `speedrun.sh`): `--nproc_per_node=1` (all 4 instances), `--depth=12`, no `--fp8`, `--device-batch-size=4`.
- Result: 934 steps, ~80 min, final loss 1.15, ChatCORE 0.0847, HumanEval 9.15%, MMLU ~30.6%. 286M params (depth 12, n_head=6, n_embd=768).
- Confirmed working via `python -m scripts.chat_cli` (inside activated nanochat venv) — coherent English structure, factually unreliable, as expected at this scale.
- Both checkpoints backed up under `/workspace/nanochat_checkpoints/`:
  - `chatsft_checkpoints/d12` — the checkpoint AFTER nanochat's own generic SFT pass. This is the one `chat_cli` loaded by default and the one verified working above. NOT the Phase 3 starting point — nanochat's generic SFT is not my hand-implemented SFT.
  - `base_checkpoints/d12` — the raw PRE-SFT checkpoint, straight out of pretraining. **This is the actual Phase 3 starting point.** Verify this path/files exist (model, meta, optim, same pattern as chatsft) before beginning Phase 3 — confirmed present as of the last check, but re-verify if picking this up much later.

**Exit criterion met:** model loads and generates text on my own infrastructure, both checkpoints safely stored, correct one identified for Phase 3.

## Phase 3 — Supervised Fine-Tuning (SFT) — NEXT UP
**Goal:** hand-implement a fine-tuning loop (not using nanochat's built-in chat_sft) that takes the base checkpoint and trains it to reliably produce summaries on request.

- **Read:** InstructGPT paper, SFT section — Ouyang et al., 2022 — https://arxiv.org/abs/2203.02155
- **Data:** TL;DR SFT dataset (human-written reference summaries) — https://huggingface.co/datasets/vwxyzjn/summarize_from_feedback_tldr_3_filtered
- **Code location:** `/workspace/VerdictGPT` (my own repo) — NOT inside nanochat.
- **Starting point:** load the base checkpoint from `/workspace/nanochat_checkpoints/base_checkpoints/d12` (the pre-SFT base model — confirm this exact path/tag exists before assuming; nanochat's own SFT pass created a separate `chatsft_checkpoints/d12`, don't confuse the two).
- **What "done" looks like:** a model that, given a passage, reliably outputs a concise summary rather than free-associating text — measurably different from base model behavior on the same input.
- **Approach:** write the training loop myself (with Claude Code explaining syntax/mechanics as tutor), covering: loading the base model, formatting the TL;DR data into prompt/target pairs, a standard supervised loss (next-token prediction on the target summary), and a training loop that updates the model.

## Phase 4 — Reward model — NOT STARTED
**Goal:** hand-implement a model that scores a (prompt, summary) pair for quality, validated against real human preference data.

- **Read:** InstructGPT paper, reward modeling section (same link as Phase 3).
- **Reference code to study structure against (not copy):** `ashworks1706/rlhf-from-scratch` — https://github.com/ashworks1706/rlhf-from-scratch
- **Data:** TL;DR preference-pair dataset (human rankings of which summary is better) — https://huggingface.co/datasets/openai/summarize_from_feedback
- **Code location:** `/workspace/VerdictGPT`
- **What "done" looks like:** given two summaries of the same text, the reward model's score agrees with the held-out human preference label at a rate meaningfully better than chance.

## Phase 5 — PPO — NOT STARTED (the hard part)
**Goal:** hand-implement the full RL loop that uses the reward model to improve the SFT model.

- **Read:** original PPO paper — Schulman et al., 2017 — https://arxiv.org/abs/1707.06347
- **Read (ongoing debugging manual, not one-time):** "The N+ Implementation Details of RLHF with PPO" — https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo
- **Study after my own first attempt, to check work, not copy:** `vwxyzjn/summarize_from_feedback_details` — https://github.com/vwxyzjn/summarize_from_feedback_details — and `huggingface/trl` — https://github.com/huggingface/trl
- **Build:** full loop coordinating four models — policy (being trained), frozen reference copy, reward model, value model. Log reward score and KL-divergence to W&B from the very first run.
- **Expected failure mode, not a red flag:** reward spikes, KL-divergence explodes, output degrades into repetitive nonsense ("policy collapse"). This is genuinely useful material for the eventual writeup, not something to hide.
- **Fallback if PPO stalls out completely:** DPO (Direct Preference Optimization) — simpler, related, a deliberate documented pivot rather than a dead end.
- **Optional deeper reference:** Stanford CS336, "Language Modeling from Scratch" — https://cs336.stanford.edu/
- **What "done" looks like:** the RLHF-trained model produces summaries that score measurably higher on the reward model than the SFT-only model, without collapsing into degenerate output.

## Phase 6 — Demo and writeup — NOT STARTED
- Build the three-way comparison (base → SFT → RLHF) on identical inputs, using the React/Vite/Vercel stack.
- Turn `NOTES.md` into a short writeup: what was built, what broke, what was learned — this is the actual deliverable for university applications/resume, not just the code.
- Clean up the repo: clear README, sensible structure, link to the live demo.
- Record a short demo video as backup in case a live demo isn't practical in an interview setting.
