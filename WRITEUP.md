# VerdictGPT: Building RLHF From Scratch

## What this is

A hand-built implementation of the full RLHF pipeline — supervised fine-tuning, a
reward model, and PPO — built without using any library that abstracts the core
algorithms away. Built to demonstrate real, working understanding of how modern
language model alignment actually works, not just how to call an API.

Standalone project, unrelated to any employer or coursework. Built solo, on a
single rented RTX 4090.

## The pipeline

**Phase 0-1 — Foundations.** Studied transformer internals (attention, causal
masking, positional encoding) via Karpathy's "Let's Build GPT," then read
nanochat's own source code line by line with an AI tutor explaining syntax,
since I had no prior software engineering background.

**Phase 2 — Base model pretraining.** Trained a GPT-style model from scratch on
general web text. First attempt: 286M parameters (depth 12). This model worked,
but had a clear failure mode on longer, open-ended generations — it would fall
into short repetition loops ("the the the..."), a known symptom of an
undertrained model at that scale. After exhausting fine-tuning experiments on
top of it (see Phase 3), retrained a larger, 897M-parameter version (depth 20),
which fixed the repetition problem and produced noticeably more coherent,
factually grounded text.

**Phase 3 — Supervised fine-tuning (SFT).** Fine-tuned the base model on
human-written summaries (TL;DR dataset) to teach it the specific task of
summarization. Ran a systematic series of experiments varying data shuffling,
step count, and batch size (via gradient accumulation) to find what actually
improved output quality. Batch size was the single most reliable lever — larger
effective batches consistently reduced training noise and improved results,
across both the 286M and 897M models. Confirmed a real, evidence-backed
finding: the smaller model's ceiling on content accuracy (correctly identifying
who did what to whom in a summary) was a capacity limitation, not a training
procedure problem — the larger model measurably fixed several of these errors.

**Phase 4 — Reward modeling.** Built a model that scores a single summary,
trained on ~90k human preference pairs (which summary a person preferred, of
two options). Tested multiple learning rates, batch sizes, and head
architectures. Settled around a stable 56-58% accuracy at predicting human
preference — real, above-chance signal, but modest, and honestly reported as
such rather than oversold. This is meaningfully below large-scale published
reward models (65-75%+), consistent with the much smaller scale of this project.

**Phase 5 — PPO.** The hardest and most fragile phase. Implemented all four
required models (policy, frozen reference, reward, value) and the core PPO
mechanics: reward scoring, KL-divergence penalty against the reference model,
advantage estimation via the value model, and the clipped policy update.

Hit and diagnosed several real bugs along the way, most notably a dataset
indexing bug that caused the model to train on the same post repeated four
times in a row — which looked exactly like catastrophic model collapse until
traced back to its actual, mundane cause. Also investigated an apparent
reward-hacking pattern (the model injecting aggressive language into outputs)
before determining, through direct side-by-side testing, that this behavior was
already present in the base model's pretraining data, not something PPO
specifically introduced.

## Final result

A three-way, side-by-side comparison of the base model, the SFT model, and the
PPO model on the same held-out inputs. Across three real test cases, the PPO
model was the most consistently coherent of the three — the SFT model
occasionally collapsed into pure repetition on certain inputs, a failure the
PPO model did not exhibit on the same inputs.

## Honest limitations

- The reward model's ~57% preference accuracy is a real, acknowledged
  ceiling at this scale — not fixed despite testing three learning rates,
  two batch sizes, and two head architectures.
- PPO here uses a per-sequence (not per-token) reward and ratio calculation —
  a legitimate simplification, not the full per-token credit assignment used
  in state-of-the-art implementations.
- Both models still occasionally invert factual details in a summary (e.g.,
  who is jealous of whom) — reduced by the larger base model, not eliminated.
- Compute constraints (a single consumer GPU, a real dollar budget) shaped
  every scale decision in this project — model size, training length, and
  batch size were all chosen under real, stated cost tradeoffs, not
  arbitrarily.

## Infrastructure lessons (the unglamorous half of the project)

A large share of the real engineering effort in this project was not
algorithmic — it was infrastructure: recovering from repeated disk-quota
crashes, discovering that a script was silently duplicating checkpoints to an
unsafe location, fixing a corrupted-file resume bug, and diagnosing a
data-loading bug that looked identical to a training instability bug. Every
long-running script in this project was eventually built with checkpoint
saving, resume support, and automatic old-checkpoint cleanup — none of which
existed in the first version of any of them.
