# VerdictGPT

A hand-built RLHF pipeline (SFT, reward modeling, PPO) trained from scratch
on a nanochat base model. See WRITEUP.md for the full project summary,
results, and honest limitations.

## Structure

- `phase3_sft/` — supervised fine-tuning scripts and experiments
- `phase4_reward_model/` — reward model training
- `phase5_ppo/` — PPO implementation
- `demo/` — final three-way comparison (base vs SFT vs PPO)
- `NOTES.md` — raw working notes from development
- `WRITEUP.md` — the actual project summary
