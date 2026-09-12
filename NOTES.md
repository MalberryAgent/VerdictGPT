# Notes for project creation

### Contains:
- Changes made
- Learning curves
- Progress timeline
- Challenges

## Notes


___________________________________________


## August 22 - (4hr)
Got repo dialed and connected, then started and connected RunPod, created pod gpu and connected through ssh, downloaded packages, general set up stuff. When running pip install for packages, i ran into roadblock bug, and found i was not in the repo folder, so had to do cd /VerdictGPT, then added the file to it, so that i can downlaod all the packages on that file, so note, you cant download packages, etc, in a directory, you have to make a file, eg, requirements.txt.
#### Package Versions (future reference)
- Name: torch - 
Version: 2.8.0+cu128
- Name: transformers - 
Version: 5.15.1
- Name: datasets - 
Version: 5.0.1
- Name: wandb - 
Version: 0.28.2


__________________________________________


## August 23 - (4hr)
Watching first vid tutorial!
### Let’s build ChatGPT: From scratch, in code, , spelled out. - Andrej Karpathy

ChatGPT is transformer architecture as written in a research paper in 2017 “Attention is all you need” and implemented in GPTs models.
Tutorial: Build a pre trained GPT from scratch and using package to inspect and adjust internal mechanics at the code level, (CHECK OUT NANOGPT REPO)

Pre training is training a neural network on massive web datasets to predict the next token, producing a document completer, not assistant, then fine tune base model with QA written by human, then RLHF (Reinforcement Learning from Human Feedback) to align model, and reward model.

Begins with empty file, define  transformer, then train on Shakespeare dataset, and generate Shakespeare infinitely.

words of Shakespeare text as set, and list on top to create order and meaning, and then it sorts.

Strategy to tokenize - take raw string of text to integers to be vocab for possible  elements, characters -> Integers

Hi there = [41, 56, 23, 67, 23, 67, 46, 15] You can encode and decode, this is one of many possible methods. This is character level tokenizer.

Tokenize library of Shakespeare, all the text, is a very large sequence of integers.
To improve efficiency and future proof, save 90% of library to train, then remaining 10% to validate.

Train transformer - 
Train by chunks with maximum length, not all at once because then to much compute.

Think of prediction like this:
Hi there = [18, 47, 56, 87, 13, 45]

In context of 18, 47 likely comes next. In context of 18, 47, the next is likely 56. In context of 18, 47, 56, then 87 is likely next.
5 Examples hidden in a chunk of 6.

A data loader prevents memory crashes by chopping a giant text library into small snippets called chunks. It then groups these chunks into batches so the computer can learn from multiple snippets at the exact same time, handing and structuring for the ai training.

##### Building simplest AI possible - Bigram Model.
Tries to predict next letter only looking at the one before, like seeing Q, and guessing U. Measuring loss track progress, its like a mistake score, the Lowe the number, the better its guessing. SO an untrained model would have a high loss and it would be gibberish. This is the structure to set up. AI can guess the next letter, measure how wrong then generate text.
Next, training the bigram model by running it through thousands of training loops to teach it using a tool called an optimizer. The code grabs text, measure the loss, calculates what number caused error, then nudges in the right direction slightly, gibberish turns into English like word patterns.

### VS Code structure so far: (so whole program runs automatically form start to finish)
- Read data
- Encoder/decoder
- Split strings into batches
- Data loader to get batches and targets
- The bigram language model (logits:loss:generate)
- Optimizer
- Training loop

Output in terminal:
Train loss, verify loss, and showing lower loss (good). Then the sample it generated pre-training

To better calculate loss, get the mean throughout batches to get an average combined loss. Because right now, the loss is split every batch, and cannot be translated across training.

### BTC - 3d shape of the tensor (number grid) moving through the model. It defines how data is packaged as the ai learned to look into the past to learn and improve.
- B = Batches - 4 independant text snippets running at same time.
- T = Time/context - The sequence length, 8 characters in a row inside each snippet.
- C = Information/Features - Amount of information/numbers stored  for each individual character position.

Imagine a classroom with 4 rows of desks (B), with 8 students/row (T), and each student holds an index card with 2 numbers written on it (C).

Why does it matter to the model? A simple bigram model is short sighted, so it only looks at the current character to predict the next, so to improve it needs to know what character 1-4 said before it guesses 5.

#### Next, Karpathy, replaces the slow step by step python loop shown above with a math shortcut called matrix multiplacation. Instead of doing an order of operations, PyTorch calculates the running average of all the past characters at once, this trick changes inefficient manual looping, into a fast, efficient parallel operation for GPUs

Before, calculating what came before needed slow code loops to measure character 1, then character 1 and 2, then 1, 2, 3, one by one. Now, instead of calculating manually, the trick creates a “triangular template” made of fractions.

Row 1 - [1, 0, 0] - keeps all of character 1.
Row 2 - [0.5, 0.5, 0] = averages of character 1, 2.
Row 3 - [0.33, 0.33, 0.33] - averages character 1, 2, 3.

^all rows sum to 1, therefore are averages

Pressing the triangular template onto the dataset using matrix multiplactiona, calculates every single running average across the entire text in one instruction.

### Dictionary -> Weighted Aggregation:
Gathering multiple pieces of information together and giving specific pieces more or less importance (weight) depending on how relevant they are.

^ = Version 1


### Version 1 (The Hand Calculation): Imagine an accountant taking a stack of paper forms and manually calculating running averages using a handheld calculator, one sheet at a time, row by row.

Version 2 (The Automated Stamp): Instead of calculating each line by hand, you build a custom rubber stencil (wei). You press this stencil onto the entire stack of paper all at once using a high-speed machine (the GPU). The machine stamps all batches instantly and outputs the identical numbers in a fraction of a second.

### Next, Version 3:
In version 2, he sends future spots to 0, but real transformers need  a more flexible way to score how important past words are relative to each other using “Softmax”. Softmax is like a machine that takes points, and converts them into a pie chart (percentages add to 100). When the softmax machine sees a score of -infinite, it gives that spot 0% share of the pie. The AI is left with clean percentages for past words while future words remain hidden. (-Inf = a function that turns raw scores into percentages. To block the AI from peeking at future characters, he replaces future spots with -inf (negative infinity). Passing -inf through softmax forces those future positions to an exact 0% probability. This leaves only the past characters converted into clean percentages that sum to 100%, creating the flexible foundation needed for self-attention. It basically blocks the ai from looking into the future and makes it focus on the past, to predict the future.

### Dictionary - softmax:
 A standard mathematical function that takes any list of raw point scores and converts them into positive percentages that add up to 1.0 (100%)




_____________________________________





 ## August 25 (3hr)

### Karpath Video Cont'd

### ATTENTION is…
Instead of evert word standing alone, each word looks at past words and asks “who has information that helps me make sense”. Attention gofers in the model, in the transformer blocks, between reading the raw input words and making the next-word prediction. The data stream, is how information flows form past words to current words, so attention only looks backwards so the model can’t accidentally cheat by looking at future words (-inf). Attention improves context and meaning, solving multi meaning words, like knowing if “bank” means riverbank, or money, it looks backwards to see if nearby words mention “water” or “money”. Also, attention does not forget details, it allows a word at the end of the chapter to directly link back to a word mentioned on page one. In code, it looks like a matrix of percentage scores from 0 to 1.0. Visually, it would looks like a heat map grid, where if row 5 is the word “it”, and column 2 is robot, a bright square (85%) shows the modcel is paying lots of attention to robot, to understand it.

#### Shows how transformer models scale their attention scores by dividing by a function before passing to softmax. This scaling trick keeps numbers at a manageable size so the model can actually learn.


Imagine a room where 64 people are whispering opinions to you. Because everyone speaks softly, you can weigh all their thoughts together. But if you pass all their voices through a massive amplifier, the slightly loudest voice becomes a deafening roar while everyone else gets drowned out completely. You end up listening to only one person. Scaling down attention scores keeps everyone at a balanced conversational level so the model can aggregate insights from multiple words at once.

With the scaling division (Balanced): Word A gets 25%, Word B gets 60%, Word C gets 15%. The AI listens to a blend of all three words.


#### Next -> Multi head attention
next step after scaling attention scores is implementing multi head attention, running multiple self-attention heads in parallel so the model can process multiple context relationships simultaneously.
Think of instead of one person taking everything at once, where only one detail can be focused one at a time, or many can be lesser quality results. Multi head attention  is like a panel of specialized experts to focus individually, in paralell on different parts of the sentence to fine tune the greatest results for improving and scaling the model by attention.
If each of the 4 experts produces an 8-number summary, stapling those summaries side-by-side creates a single 32-number master report containing every perspective. Channel Division: Splitting total embedding dimension across heads. If total channels $C = 32$ and you use $4$ heads, each head operates on a head_size of $8$ ($32 \div 4 = 8$).
Why Multi-Head Attention Matters A single head can only focus on one primary type of relationship at a time. By splitting feature space across multiple smaller heads, the model learns a vastly richer understanding of sentence structure without increasing total computational cost.

After implementing multi head attention, the next step is called “Feed Forward Network” which gives each token a dedicated time to think about the information it just gathered. If you take this to an analogy, its like all the experts form multi headed attention, get to go back to there private desks to digest what they learned and gathered without interacting with anyone else. Then the workflow consistss of 3 main parts, Linear layer one, like spreading the notes on a large whiteboard to explore complex idea and connections. Then activation function, the worker erases useless ideas keeping only the valuable ones, the linear layer 2, where the worker condenses the valuable stuff back down into a clear report format to prep for next round of meetings.
Why the Feed-Forward Network Matters:
Attention only shuffles and mixes existing information across tokens—it doesn't actually do heavy processing on its own. The Feed-Forward network acts as the brainpower where the model reflects on those gathered facts and learns deeper facts about the language.

Communication vs. Computation: Self-attention handles communication (tokens looking at other tokens). Feed-forward handles computation (tokens processing their own data internally).

The next structural upgrade is adding Residual Connection giving the network a direct "superhighway" to pass information straight through without getting lost as the network gets deeper.

the next key component is adding LayerNorm (Layer Normalization)—a technique that continuously re-calibrates the numbers moving through the network so they don't explode or shrink as models grow deeper.

Then training ran and outputted decent shaksepare, after 15 minutes of training on tip of the existing code.

#### Encoder/Decoder frameworks

Encoder-Only (The Document Reader / BERT): Imagine someone reading an entire document at once. They can look forward and backward across every sentence to understand the full context. Great for summarizing, search, or sentiment analysis, but bad at generating new text word-by-word.
Decoder-Only (The Live Storyteller / GPT): Imagine an author writing a novel line-by-line. They can look back at what they’ve written so far, but they cannot see into the future. They use a lower-triangular mask (causal attention) so future words are hidden, predicting the next word auto-regressively.
Encoder-Decoder (The Translation Team / T5 / Original Transformer): Imagine a two-person translation team. The Encoder reads an entire English sentence at once and writes down a summary note. The Decoder reads that note and generates the French translation one word at a time, using cross-attention to look back at the summary note while writing.

### Dictionary:
Variance: A measurement of how widely spread out a group of numbers is from their average.\

Peaky: When a probability distribution gets squished so hard that one option shoots up to nearly $1.0$ ($100\%$) while all others drop near $0\%$, turning a subtle comparison into an absolute "winner-take-all" outcome.

Query (q): A word asking a question (e.g., "Where is the action in this sentence?").

Key (k): A word advertising its contents (e.g., "I am a verb!")


### A little summary, and how ChatGPT works, and how it becomes a conversationsal assistant

1. Pre training - The model learned grammar, facts and world knowledge bu predicting the next token on massive amounts of internet text. But it’s only a continuation engine, so if you ask “what is the capital of France”, it may respond “What is the capital of Germany” because it thinks it is completing a quiz writeup.
2. Supervised fine tuning - Humans with thousands of prompt and response dialogues with prompts, and the correct/expected answers. The Mosel is trained on that structured dataset.
3. RLHF (Reinforcement learning from human feedback - Reward modelling, it generates multiple candidate responses for one prompt, the humans rank them from best to words, and a separate reward model tries to predict the humans scores : Optimization is when reinforcement learning algorithms tweak the language models parameters so it consistently outputs the responses that receive high score from the reward model.



 ## Video Overview:
 follows a modular pipeline: converting raw text into numbers, building self-attention mechanisms to learn word relationships, stacking those layers into Transformer blocks, and training the network auto-regressively.
 
Starts with data preparation and tokenizations, where the encoder takes every character in the dataset and turns it into a unique integer (ID), Then the text is cut into short snippets  and then multiple of those snippets are stacked into parallel groups so the computer can process as any examples in parallel. Before building the complex attention logic, a basic baseline model only looks at the single current character, and guesses what comes next based on statistics, it ignores the surrounding context. Then single head attention, there are 3 roles, (Query, Key,Value) and every character is given 3 roles: Q:what am I searching for, K: what information do I hold, V: What message do I pass along. Then the model compare these to calculate how relevant past characters are to the current one, and a rule is that (-inf) blocks characters from seeing into the future (no cheating!). The relevency gets scored into percentages, which pull I weighted amount of information from past characters. Now the next layer of attention, multi head. Instead of relying on once reader the model splits into several smaller heads working in parallel. One might do grammar rules, while the other tracks themes. And their findings get glued back together in the end. On top of attention, each character passes through its own private calculation step, so its like individual thinking time to digest what it just learned  from other characters before moving forward, this is called Feed Forward Network. Then blocks stack and training must be kept stable to improve the model. First, transformer blocks, where the attention dn reflection step go into one single block, and multiple of those blocks get stacked on top of each other to make the model smarter. Then original data bypasses each step and gets added to the output, to create a direct route for feedback so the model won’t forget early information, like a highway. Then normalization layers continuously keep calculations at a manageable size so the number dont blow up, and become innacurate. Finally it’s time for predicting and generating new text. The final numbers are converted into percentage changes for eery character in the dictionary, to move toward the final output, and finally the auto-regressive loop. The model picks the most likely next character, pastes it onto the end of the text, and feeds the new longer string into itself to predict the next letter, generating complete sentence one character at a time.

### DONE VIDEO

_____________________________



## August 26 (3hr)

### Nanochet repo study

### Nanochat/gpt.py digest
I notice the first couple sections are “preparing”, is the foundation and configuration before anything gets done, it lets the things mentioned later in the important code exist. Like importing libraries like torch to do the neural network math, taking contents from other files that should be taken into consideration, and even listing settings and instructions for developers as context. 

Think of coding as a recipe book with a strict grammar. All it is doing is giving a machine step-by-step instructions to turn inputs into actions.

#### These are the answer to the questions I had, plus some more notes to understand this 1st part of the gpt.py digest. ->
#### Class
Blueprint for making identical copies of things, like a recipe, so when It went class CasualSalfAttention, all its doing is saying, this is what casualselfattention looks like, everytime someone creates one, here’s what goes inside it. No settings, it’s a template, the settings only exist when you create an instance from the template.
####  _init_
This is the inittialization function, the tings that runs every time you create a  new copy from the template, it’s the moment the baker is cooking the recipe. So when you make a new casualselfattention, python will auto run init, and a bunch of other assignments happened (self.n_head = …), which create that specific copies internal stuff. 
#### Self
Eg. (self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd //
)
Self means, this specific copy im talking about right now, when you write self.n_head = config.n_head, your saying “for this specific copy, store the value config.n_head inside a slot called n_head”, later when you use self.n_head in the code, your saying give me n_head for this specific copy. Have cookies made from same recipe, each one is own cookie (self), when you bite into this cookie and it’s sweet, that’s because of self.

In __init__:
self.n_head = 6  # "This particular attention layer has 6 heads"

Later in forward():
for each_head in range(self.n_head):  # "Use the 6 heads from THIS layer"




#### I read it but dont have much of a background, so getting a digest going through everything In there, mainly about the attention part of the code.

#### super()._init_()?
super() means: call the parents version of this function. Its like before I run my own stuff, run this setup code first.

#### What does nn.?? Mean
Sort for torch.nn its the lib inside PyTorch which does all the math like matrices, etc.

#### How does QKV fit into everything.
You have concept of each one individually, but it’s like, what am I, name tag, and info it gets.

#### Attention process
Each token has QKV.
Each token with Q asks which tokens in my past have info I need, It compares its Q against all the past tokens K’s, asking do your nametags answer my questions. The token with the best matching Ks say yes, here is my value and then hand over there V vectors. The current token blends together all those values, weighted by how well did you K match my Q into one final message.

q = self.c_q(x)  # Turn current embedding into a "question"
k = self.c_k(x)  # Turn embeddings into "nametags"
v = self.c_v(x)  # Prepare the "answers" to share


### Code definition: kv_cache:
if kv_cache is None:
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
else:
    k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
    y = flash_attn.flash_attn_with_kvcache(...)
    if self.layer_idx == kv_cache.n_layers - 1:
        kv_cache.advance(T)
Two paths — two situations:
Path 1: Training (if kv_cache is None:)
We have the whole sequence available. Run attention normally: compare this token's Q against all past Ks and Vs.
Path 2: Generating one token at a time (else:)
We're generating text word by word. We already computed K/V for all previous tokens in earlier steps. Don't recompute them — just grab them from the cache.

k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
"Hey cache, give me the K and V you saved from before for this layer."

y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"Run attention: compare NEW token's Q against CACHED old K/V, plus this new token's fresh K/V."

if self.layer_idx == kv_cache.n_layers - 1:
    kv_cache.advance(T)
"We're done with all layers. Tell the cache: 'Remember THIS token for next time.'"


### Im finding that a lot of code is one of these. Setting up something, giving instructions, telling what other code to do, defining code, telling what computer does. It’s either interacting with itself or the computer. Its like a guy telling someone how to do something, asking for something, or giving instructions.


### GPT _init_ - Building the entire model

Some notes: Embeddings have layers, right now, an embedding that takes the integers that were once the characters, and converts them into vectors.
 
### Generate - making the model product text
"The model picks the most likely next character, pastes it onto the end of the text, and feeds the new longer string into itself to predict the next letter, generating complete sentence one character at a time."
 First code solidifies it’s not training, it’s generate, “We're not training, so don't compute gradients. Turn off all training machinery.”

Start: ids = [5, 42, 13]  (3 tokens)

Iteration 1:
  forward(ids) → logits for next token
  Pick token 7 (based on scores + sampling)
  ids = [5, 42, 13, 7]
  Yield: 7

Iteration 2:
  forward(ids) → logits for next token
  Pick token 99
  ids = [5, 42, 13, 7, 99]
  Yield: 99

Iteration 3:
  forward(ids) → logits for next token
  Pick token 42
  ids = [5, 42, 13, 7, 99, 42]
  Yield: 42

Run loop max token times ^^^

### gpt.py in a paragraph
The file is essentially a blueprint for a machine that reads tokens (numbers representing words) and predicts what comes next. Think of it like teaching someone to write by showing them thousands of examples, until they learn patterns.
The file starts by defining the settings: how many layers deep, how wide each layer is, how many vocabulary words exist. Then it builds the actual machine piece by piece.
First come the building blocks. There's an attention layer, which is like a student reading a sentence and asking, "which words from before help me understand this current word?" It can look back at the past, gather relevant information, and blend it together. Then there's a thinking layer (MLP), like private desk time where each word processes what it just learned without talking to others. A Block combines both: listen to context, then think independently.
The model stacks twelve of these Blocks on top of each other. A token enters as a number, gets converted to a 768-dimensional vector (think of it as a rich description of that word), flows through all twelve Blocks getting refined each time, then comes out the other end as scores predicting every possible next token.
When training, the model sees the correct answer and measures how wrong it was. When generating, it picks tokens one at a time based on those scores, adds them to the sequence, and runs the whole thing again, building text token by token.
That's the journey: a token comes in empty-handed, flows through twelve layers of listening and thinking, exits with deep understanding, and tells the caller which word probably comes next. The model repeats this thousands of times during training until it learns language, or generates text by repeatedly asking "what's next?" until a full response is built.


## August 27 (1hr)

### Tokenizer.py digest

strategy to tokenize — take raw string of text to integers to be vocab for possible elements, characters → integers. This file does exactly that. But instead of converting each character individually (too many numbers), it converts chunks of characters (words, subwords, numbers) into unique integers. It's like having a dictionary where each entry is not a single letter, but a meaningful piece of text. Then when the model reads text, it looks up those chunks and converts them into numbers. 

What's a tokenizer?
A tokenizer is a machine that converts text ↔ numbers.
* Text → Numbers (encode): "hello" → [1000, 2034]
* Numbers → Text (decode): [1000, 2034] → "hello"
Then a process runs:Like you're building a dictionary. You read a huge library, count which words/phrases appear most, and decide to include the top 32,760 in your dictionary. Then you print that dictionary into a book (tiktoken encoding) for fast lookup.

#### Syntax is:
Think of it as the grammar rules of the language. If you break these rules, Python will throw a SyntaxError and refuse to run your code.

_____

Process

"Hi there = [41, 56, 23, 67, 23, 67, 46, 15]" — converting text to integers.
This file does that, plus more:
1. Defines special markers (BOS, user_start, etc.) to structure conversations
2. Trains a tokenizer by learning which text chunks are common (rustbpe)
3. Provides fast lookup to convert text ↔ numbers (tiktoken)
4. Handles conversations — converts a chat back-and-forth into tokens, marking which parts to train on
5. Supports RLHF — renders conversations for reinforcement learning (removing the target response)


### Summary of Tokenzer.py
When the model reads text, the tokenizer breaks it into chunks and converts each to a number. When the model outputs numbers, the tokenizer translates back to readable text.
For conversations, the tokenizer does something smart: it marks which parts the model should learn from. User messages get marked "don't train" (input). Assistant responses get marked "train" (the model should learn these). This focuses training on teaching the model to respond well.
It's the translator between human text and machine numbers.
__________

### scripts/base_train.py digest

Quick overview:
 training the bigram model by running it through thousands of training loops to teach it using a tool called an optimizer.This file does exactly that, but at a massive scale. It orchestrates the entire training process: loading data, building the model, setting up optimization, and then running thousands of training steps where the model reads text, predicts the next token, measures how wrong it was (loss), and nudges its weights in the right direction (backprop + optimizer).


* --depth — how many transformer blocks (12, 20, 50, etc.)
* --device-batch-size — how many sequences to process at once on one GPU
* --total-batch-size — total sequences across all GPUs (determines how many gradient accumulation steps)
* --matrix-lr, --embedding-lr — learning rates for different parameter types
* --eval-every — how often to evaluate validation loss
* --sample-every — how often to generate samples to see what the model learned

Summary:
The file is the training orchestrator. It sets up command-line arguments for every aspect of training (model size, batch size, learning rates), then runs the core training loop thousands of times.
It detects your hardware (GPU/CPU), loads the tokenizer, builds the model, and initializes it with random weights. Using scaling laws research, it automatically calculates optimal batch sizes and learning rates based on model size.
Then comes the loop: load a batch of text, run through the model (forward pass), compare predictions to ground truth (loss), compute gradients (backward pass), and update weights (optimizer step). Every few iterations, it evaluates on unseen text, generates samples to see if learning is working, and saves checkpoints.
The script runs thousands of times, watching loss decrease as the model learns. By the end, you have a trained model.

________

### Optim.py digest

It reads the gradients (which tell you "how much did this weight contribute to the error?") and carefully adjusts each weight to reduce that error. This file contains two optimizers: AdamW (for embeddings and scalars) and Muon (for matrix weights), combined into one.

Your notes describe the training loop: "The code grabs text, measure the loss, calculates what number caused error, then nudges in the right direction slightly."
This file provides the "nudge":
1. AdamW nudges embeddings and scalars using momentum + adaptive learning rates
2. Muon nudges weight matrices using orthogonalization (rotation-based updates)
3. Distributed sync ensures all GPUs agree on the updates
4. Async communication overlaps with computation for speed
The model gets nudged thousands of times, and gradually, loss decreases.

Short Paragraph (150 words):
The Journey of optim.py
The file contains two optimizers working together. AdamW nudges embeddings and scalars using momentum (remembering past directions) and adaptive learning rates (moving faster or slower based on gradient magnitude). Muon nudges matrix weights differently — instead of moving arbitrarily, it finds the nearest "rotation" (orthogonal matrix) to apply, based on research showing this is more stable for training.
For distributed training on multiple GPUs, the optimizers sync gradients across devices, with large parameters split across GPUs to save memory (ZeRO-2 style). Communication happens asynchronously so GPUs don't sit idle — while one GPU finishes syncing, others compute updates.
The combined MuonAdamW class orchestrates this in three phases: launch communication, compute updates, wait for synchronization. Repeat thousands of times during training, and the model learns.


## done nanochat file digest--

________

## August 28 (4.5hr)

#### Kicking off training (1st dev operation on project)
Firstly there is a script i have to run called speedrun.sh, its basically a sequnce of many terminal commands i would have to manaully input, but this way it does it at once, and i can monitor whats happening. one thing, is that i have to specify to the command because it also includes some downloading and training, and the script expects that to run on a more powerful gpu i have rented in runpod, so it will over capazatize it. So i have to specifify in a reletive command, that im using a rtx 4090, so it will adjust, but it will still take 1.5-2 hours.
Keep having to mention claude to explain more, not mention future choices because it makes it confusing on what to do. I need specific, sequenced and clear steps with mini explanations so i can learn a bit on the way.

Ok, started the training run in nanochat. its weird, i though it was going to be in Verdict, but no, theres some pre done stuff in nanochat that makes it easier, so we shall see. then i had to download some things, then did a thing were it teleports to code running in the runpod, then ran the training command and its going to run for hours, and i can close my laptop. have to check on it every now and thenSpecifically, this one script does three things in sequence:

Builds a tokenizer — the thing that turns text into numbers the model can read.
Pretrains the model — reads a large pile of web text repeatedly, learns to predict the next word, over and over, until it's a working language model. This is the part you already understand conceptually and read the code for.
Does a first SFT pass — teaches it to respond in a conversational format instead of just rambling.

End result: a model file sitting on your pod that you can actually talk to (chat_cli). It'll be rough — small, undertrained, "kindergartener" quality, as expected at this scale.

##### Its been 5 hours in training
its on step 779 out of 950 and 83% done, and the loss is down to 1.15, from 5.19, which is insane. its about 1 hour from completion. One thing I notice, is that throughout training, you expect loss to go down a tiny bit per turn, sequentionaly, but instead, its like 1.17 -> 1.22 -> 1.24 -> 1.21 -> 1.14 -> 1.28 -> 1.23 -> 1.15. Because each turn is a seperate instance, liek rinsing and repeating, and theres another instance that sees how it does, compares, it and improves it by a fraction of decimal, thats why it takes so long.

Just a mention before i move on or forget. for loggine purposes. i have depth of 12, i had to change that to 12 from 20 before training, because then the process would have demolished my GPU.

### training + base model complete!
Phase 2 complete — base model + SFT
- Architecture: depth 12, 286M params
- Training: 934 steps, 4 hours, single RTX 4090
- Final loss: 1.15, ChatCORE: 0.0847, HumanEval: 9.15%
- Sample: asked "why is the sky blue" — got grammatically coherent but 
  scientifically wrong/circular answer. Expected at this scale — proves 
  the model learned English structure without learning real facts, which 
  makes sense given tiny size and short training time.
- Checkpoint backed up to /workspace/nanochat_checkpoints

Here was the first answer of VerdictGPT:
____________________________
NanoChat Interactive Mode
--------------------------------------------------
Type 'quit' or 'exit' to end the conversation
Type 'clear' to start a new conversation
--------------------------------------------------

User: Why is the sky blue?
Assistant: The sky is indeed blue, primarily due to the way our eyes perceive the color of the Earth. Our eyes are constantly adjusting to see the blue and white of the sky, which is why we can see it on a clear day. 

The reason for this is that our brains are wired to take in the colors of the sky, and it's easy to see the blue and white of the Earth in the background. However, when we look at the sky, we're essentially seeing the color of the blue and white of the Earth, which is why we can see it. This is why the sky is blue.

Additionally, our brains are also wired to recognize the color of the Earth, which is why we can see it. The blue and white of the sky is a result of the way our eyes detect the blue and white of the Earth, which is why we can see it on a clear day.<|assistant_end|>

User: exit
Goodbye!
(nanochat) root@b869152993a9:/workspace/nanochat# 

____________________

yeyyayeyayeyayy!!!! It workeeddddd!!!

Im now handing off to another chat, because the intials ones got to much in it, its starting to hallucinate.

Holy I just had to debug cloning, and pushing requirements.txt and gitignore into this repo in terminal for like 30 mins. I ran into walls where it was taking things from other branches, then not remembering data so we had to nudge it, so now it will remember me, and they are pushed.

### Little project overview
<sub><sup>CONTEXT HANDOFF — VerdictGPT project, continuing from a previous chat (v2). PROJECT OVERVIEW: Building RLHF from scratch on a small language model — hand-implementing SFT, a reward model, and PPO myself, not using a library that abstracts it away. Standalone from my other projects (Tidespark/Haven). Purpose: university application + resume — specifically a "beyond my current ability, can explain it cold in an interview" project, not a shipped product. Demo will be a three-way comparison (base model vs SFT model vs RLHF model) on identical inputs, on a summarization task. IMPORTANT CONTEXT ABOUT ME: No formal software engineering background, don't know how to code yet. Conceptual ML understanding and coding fluency are two separate gaps — please don't assume I can read Python syntax, explain it when it comes up. Claude Code / agentic tools should stay in tutor/debugger mode for core ML work (attention, reward model, PPO) — not autopilot. Fine to use freely for boilerplate, debugging syntax, and the final demo UI. Original timeline: 2-4 weeks total for this project. Always give me actual clickable links when recommending resources. I need commands broken down very simply — which terminal window, exact copy-paste blocks, one step at a time. I've had repeated issues with typos from copy-paste (stray characters like a trailing "" or "[200~"), so if something errors oddly, my first instinct now is to retype the whole line fresh rather than edit it. ROADMAP (6 phases): Phase 0 — Setup: DONE. Phase 1 — Foundations: DONE (watched Karpathy's "Let's build GPT," took notes, passed a 6-question conceptual quiz on attention/masking/gradients; went through nanochat's gpt.py, tokenizer.py, base_train.py, optim.py with Claude Code in tutor mode). Phase 2 — Base model: DONE (see full detail below). Phase 3 — SFT: NOT STARTED — next up. Phase 4 — Reward model: NOT STARTED. Phase 5 — PPO: NOT STARTED. Phase 6 — Demo + writeup: NOT STARTED. Full roadmap with all resource links is saved in my repo as VerdictGPT_Roadmap.md — refer to that for the complete phase-by-phase resource list (papers, datasets, reference repos) if needed. INFRASTRUCTURE STATE: RunPod, Community Cloud, single RTX 4090, persistent volume mounted at /workspace. Pod IP/port can change if the pod restarts — I'll check the RunPod dashboard for current SSH connection details if needed. Two repos on the pod: /workspace/VerdictGPT (my own repo — this is where Phase 3 onward code will actually live) and /workspace/nanochat (Karpathy's reference repo, used only to generate a base model for Phase 2 — not my own work, just borrowed infrastructure). Same repos also cloned locally on my Mac for reading/browsing with Claude Code — separate copy, doesn't sync automatically with the pod. nanochat manages its own Python environment via uv in a .venv folder inside the nanochat repo — this is SEPARATE from the pod's system Python (where transformers/datasets/wandb were pip-installed earlier). To run anything in nanochat, must first: cd /workspace/nanochat && source .venv/bin/activate — prompt will show "(nanochat)" prefix when active. screen is installed on the pod — training runs should always be launched inside a screen session so they survive SSH disconnects. GitHub auth: using a Personal Access Token (not password) for git push/pull. Ran git config --global credential.helper store so it should be remembered now — if push suddenly asks for username/password again, that's the token expiring or needing to be regenerated at github.com/settings/tokens. PHASE 2 STATUS — BASE MODEL — COMPLETE: Ran nanochat's speedrun script, modified for single-GPU use. File: /workspace/nanochat/runs/speedrun_4090.sh (copy of the original speedrun.sh, NOT the original — original left untouched as reference). Changes made: all 4 instances of --nproc_per_node=8 changed to =1 (no single variable controls this, had to edit each torchrun line); --depth=24 changed to --depth=12 (first tried 20, ETA was ~30hrs, too long); removed --fp8 flag (H100-only feature, unreliable on 4090); --device-batch-size=16 reduced to 4 (24GB VRAM vs 80GB assumed by defaults). Training completed successfully: 934 steps, ~80 minutes total, final loss 1.15, ChatCORE 0.0847, HumanEval 9.15%, MMLU ~30.6%. Model: depth 12, 286M total params, n_head=6, n_embd=768. Script automatically continued past base_train into nanochat's own generic chat_sft + chat_eval stage after pretraining finished — this is NOT my project's Phase 3, just nanochat's built-in generic fine-tune. My actual Phase 3 (SFT on TL;DR summarization data, hand-implemented) still needs to happen separately. CONFIRMED WORKING: ran python -m scripts.chat_cli inside the activated nanochat venv, asked "why is the sky blue," got a grammatically coherent but scientifically wrong/circular answer. Expected and correct for this scale — proves the model learned English structure, as intended. Checkpoint location: originally saved to ~/.cache/nanochat (NOT on the persistent volume — this is nanochat's default and would NOT survive a pod termination). Backed up with cp -r ~/.cache/nanochat /workspace/nanochat_checkpoints — this backup is CONFIRMED DONE and should be verified still present before relying on it further. Phase 2 results logged in NOTES.md (final stats + sample chat_cli exchange). KNOWN GOTCHAS FROM THIS PROCESS: Terminal disconnects (laptop sleep, wifi drop, "Operation timed out" errors) happen often on this setup and kill anything not inside screen — always screen -S <name> before bash-ing a long-running script. screen sessions can get stuck showing "(Attached)" from a dead connection — fix with screen -d -r <name> to force-detach and reattach; if that fails with "Attaching from inside of screen?", try unset STY first, or just close the terminal window entirely and open a fresh one. If a screen session seems permanently stuck: screen -X -S <name> quit force-kills it cleanly; screen -wipe cleans up genuinely dead sessions from the list. pkill -f <pattern> reliably force-stops a stuck process from a second terminal window if Ctrl+C doesn't register (can happen if terminal focus isn't on the right window). nvidia-smi (checking GPU-Util % and whether a python process is listed) is the most reliable single command to check "is training actually running" — far more reliable than trying to reattach to screen and read live scrolling output, which caused a lot of confusion, so prefer this as the default status check. sed -i 's/old/new/g' file.sh is more reliable for editing scripts than nano's interactive find-replace (Ctrl+) in this terminal setup, which didn't work reliably. Git needed git config --global user.email / user.name set once before first commit would work; needed --allow-unrelated-histories on first pull since GitHub repo had initial content the pod didn't; needed git config pull.rebase false to resolve a divergent-branches error, all resolved now. When copy-pasting commands, watch for stray leading characters like "" or "[200~" getting inserted — if a command errors with something like "only one pattern can be provided" or "command not found" on something that should clearly work, retype the whole line fresh rather than debugging the typo. IMMEDIATE NEXT STEP — START PHASE 3 (SFT): read InstructGPT paper's SFT section at https://arxiv.org/abs/2203.02155, pull the TL;DR SFT dataset at https://huggingface.co/datasets/vwxyzjn/summarize_from_feedback_tldr_3_filtered, and note that this is hand-implemented code that will live in /workspace/VerdictGPT (my own repo), NOT in the nanochat repo, since nanochat's job of producing a base checkpoint is done — the starting point should be loading the base checkpoint from /workspace/nanochat_checkpoints/base_checkpoints/d12 and writing my own fine-tuning loop against the summarization data, with Claude Code in tutor mode explaining each part as we go, given I don't know how to code yet.</sup></sub>


_______________
### Reading the paper on SFT (supervised fine tuning)
Section 3.5

Describes a base model, with poorly characterized behaviour and doesn’t follow instructions properly, so it then gets trained with 3 different techniques, starting from those models, like GPT-3.
SFT:
Supervised fine tuning is simple and makes lots of sense when you dumb it down. They gave AI a stack of prompts alongside perfect answers written by human experts, ask then used supervised learning, which is where the AI guesss and answer, looks at the humans answer, compares and adjusts itself to match the human.
The analytical side: In this instance, the AI went through the entire stack of training 16 times (16 epochs). Then something called cosine learning rate decay, which is the AIs learning rate and how much it changes its mind when it makes a mistake. At the beginning it changes its mind lost which is high learning rate, and overtime it slowly slows sown the adjustments following a curve so it can fine tune. Then the residual dropout. They did 0.2 (20%). This makes the AI not cheat, by not letting it memorize past specific phrases and removing 20% of the content and blinds it so it forces the AI to learn general concepts. Hence machine LEARNING.
Then they pick the best version, which is either a separate ai judge to grade answers, a small batch of test questions the ai never saw. 
Something they found interesting in study: In studies, having same data fro to long its validation loss/test scores decline, it becomes a bad test taker as it overmemorizedu its homework. But they found that even though the math scores said the AI was overfitting, after just one round, they kept training. They found that keeping the AI in school for more rounds make it better in real practice, and got judged better, and received high human preference scores.


After 1 hour of studying: You know the history facts perfectly.
After 16 hours of studying: You start memorizing your teacher’s exact catchphrases. A mathematical tracker would say you are wasting time and "overfitting" to the teacher's vocabulary.
The Result: On exam day, because you spent so much time reading high-quality writing, your essays are beautifully written, persuasive, and perfectly formatted. The human examiner gives you an A+, ignoring the fact that you over-studied.

Then A bit more details on  building the AI judge (reward modeling/RM). They build the RM before training in order to score its answers. The final thing is reading a prompt and response,  then output a single quality score. Humans grade the answers on quality, truth and bias, and balance the scoring system so a standard human written answer is 0.
___________

## August 29 (2hr)

Starting to build up solid VerdictGPT in isolated folder
Added seperate instance in terminal (.venv) so its isolated, downaloaded packages, transformers, torch, datasets.
Hit 2 big problems related to the gpu nd network volume, so it wasnt loading the dataset. I only had 50gb provisioned of storage, so i updated it to 100 after a bit of detecting the problem. I found in the end that the smaller NV matched the ID of the migrated pod, so added volume. After that, ran the command and dataset finally loaded:
- 116,722 train examples, 6,447 validation, 6,553 test
- Structure: each example has `post` (long Reddit post, the input) and 
  `summary` (short human-written summary, the target) -- this is the 
  prompt/target pair shape Phase 3 training will use

_____
## August 31 (2hr)

#### Finally done phase 3, moving to phase 4, which is about checkpoint loading.
My model is a massive recipe with 300 million parameters/knobs. training is the process of slowly and percicley turning every knob to better values.

Time to write the checkpoint loading file myself, through terminal. here are the notes about code and progress:
Had to take parts from nanochats repo borrowing a single function, specifies specific parts instructions, like where to take from, etc.
pip install = downloading pre configured package.
#### Base checkpoint loading -- working

What: Wrote load_base_model.py, reused nanochat's own load_model() 
function instead of rewriting checkpoint loading myself.

Problems hit + fixes:
- nanochat not importable -> not a real pip package, used PYTHONPATH instead
- rustbpe missing -> just pip install rustbpe (not a Rust build, despite 
  old tutorials saying so)
- tiktoken missing -> pip install tiktoken

Env vars needed every session (4 total):
HF_HOME, HF_HUB_DISABLE_XET, NANOCHAT_BASE_DIR, PYTHONPATH
-> should turn into one setup script soon

Result: base checkpoint (step 1680, pre-SFT) loads successfully from my 
own code.
_
#### Base model generation test -- repetition loop found

What: tested load_base_model.py with open-ended prompt ("The best way to 
learn a new skill is..."), 30 tokens.

Result: model gets stuck in short repetition loops ("is is is...", 
"the the the..."), even after trying top_k=50 and temperature=1.5.

Why: base checkpoint only trained 1680 steps, small for a 286M param model. 
Repetition loops are a known failure mode of undertrained language models, 
not a bug in the loading code.

Why this is still fine for Phase 3: SFT fine-tunes on a narrower, more 
structured task (post -> summary), not open-ended continuation. Reasonable 
to expect meaningfully better behavior after fine-tuning, even with this 
base model quality.


### Key concepts:
Backpropagation is how an AI adjusts its internal settings after making a mistake.
If backpropagation finds the mistakes, the optimizer is the worker that actually turns the knobs to fix them.
A tokenizer is the translator between human words and AI numbers.
Overfitting happens when an AI memorizes its practice data instead of actually learning the underlying patterns.
A checkpoint is a snapshot file stored on your computer that saves the exact positions of all the AI's internal math knobs (weights) at a specific moment during training.
A gradient is just a measurement of slope—it tells you which direction error is increasing, and how steeply.

_________
## September 1 (5 hr)

#### Phase 3 SFT: Data format and training loop build:
Use nanochat convo format instead of hallucinated template, and returns ids, masking marks whihc tokens actually count towards loss. 
##### Learning rate debugging
First trained test, loss rose steadily instead of dropping, it was because the matric was scaled down at default for huge batch, this was fixed by scaling all four down. But still slowly lowered after that, it was because it was missing gradient asumalaiton, was taking 1 optimizer step per example, which is noisy. fixed by accumalating gradients over 8 examples before each optimzer step, matching nanochats approach. Then loss dropped cleanly.

#### First real test gen:
Loaded best small run checkpoint, got summary on unseen validation example. I was short and on topic, but hallucinated and incorrect. 
#### Decicsion to comence big trianing run. 
2 epocks of training (2300000 steps), targeting real quality, not just demo. Estimate 30 hours. Added resume support, so check for checkooints loads things, so in case of crash, not starting from nothing again.
Then disaster striks, it crashed, because the network volume and compute was tuned too low. Fixed with new resume of checkpoint, delete previous one right away.
Found the screen isolates from any exterior problems, as long as the pod is running, its working. By step 180k, training was healthy. Then failed again.
Conducted experiemnts, checking how quality improved over different parameters. Then after 9 experiments, concluded that 286M (12 depth) models content accuracy ceiling is likely a capacity limit, not training procedure problem. Decided to retrain the base model more and bigger (900M - 20 Depth) raher than keep tuning SFT. Cost 30 hours, 3320, total steps. its going alot, taking 1900 mins, and 30k tok/sec which is crazy.
#### Depth=20 retrain — hit repeated infra issues, all fixed

Crash 1: accidentally typed text into the training terminal while attached, 
killed the process. Lesson: never type/paste into a live training screen 
session unless it's an intentional command.

Crash 2 (same run, after fixing crash 1): resumed, hit a chain of missing 
folders one at a time — tokenizer, then checkpoint folder structure 
(needed base_checkpoints/d20, not just d20), then the dataset folder 
(base_data_climbmix) was never copied to /workspace at all. Fixed each, 
should have checked full folder structure once instead of fixing one 
piece at a time.

Crash 3: disk quota again — this script has no auto-cleanup of old 
checkpoints (unlike our SFT scripts), so checkpoints piled up over time 
and filled the volume. Deleted old depth=12 checkpoints (already tested, 
no longer needed) to free space, resized volume 100GB -> 250GB for real 
headroom, and added a small external cleanup loop (separate screen 
session) that keeps only the 2 most recent checkpoints automatically.

Each time: recovered by resuming from the last good checkpoint using 
nanochat's built-in --resume-from-step flag. One checkpoint file was 
found corrupted/truncated from a mid-crash write (same signature both 
times) — always check file sizes match before resuming from a given step, 
don't assume the last save is intact.

Status: training resumed cleanly, loss recovering normally each time 
after a brief expected spike on resume. No progress lost beyond the 
few minutes since the last checkpoint each time.

## September 2 (.5hr)

Watched video by 3blue1brown about machine learning, more context on whats going on under the hood. then checked in on trainng, and added cleanup function on another screen to delete the oldest checkpoint 5 mins after a new one is added, so no overload. also caught up writing the notes.

## Done timelaps (30hrs!)

### Starting screen recording on imovie

## September 3 (1hr)
Checked in on training every hour, bring computer wherever i went. i even checked on the bus. Cleaned up files to give room for error.

## September 4 (1.5hr)
Crash! looked in the morning, loaded credits and noticed my GPU optim was 0%, training had crashed, it had been going 800 mins, till step 200:

### Crash log:

Depth=20 training -- step 2000 eval crash, root causes found and fixed

Pattern noticed: crashes 3 and 4 both happened at the same spot -- step 
2000, during nanochat's optional CORE benchmark evaluation (fires once, 
default every 2000 steps). Not random -- same fragile piece both times.

Crash 3: eval bundle download/extraction filled up the LOCAL disk (30GB, 
separate from the 250GB network volume). Root cause: despite 
NANOCHAT_BASE_DIR being set correctly, base_checkpoints were somehow 
still ALSO saving to the old default /root/.cache/nanochat location in 
parallel (30GB duplicate, cause not fully understood, but confirmed by 
directly inspecting the folder). Fixed by deleting that local cache copy 
-- all real checkpoints were already safe on /workspace.

Crash 4: same eval step, different failure -- it found a leftover, 
corrupted eval_bundle.zip from crash 3 and tried to use it without 
re-downloading, failed reading a file that was never fully extracted.

Real fix, not another patch: disabled the CORE eval entirely with 
--core-metric-every=999999 (nanochat's own documented pattern for 
skipping it). This benchmark score isn't needed for the project -- not 
worth the repeated fragility. Permanently closes this failure category 
for the rest of the run.

Also cleaned up: deleted corrupted eval_bundle files, confirmed local 
disk back to near-empty, confirmed /root/.cache/nanochat gone.

Process change: before any relaunch now, run a full checklist (session, 
venv, env vars, checkpoint file sizes, tokenizer, dataset, local disk, 
network volume, cache clean, eval bundle clean, cleanup loop alive) 
instead of fixing one error at a time and relaunching blind. Cheaper than 
another failed multi-hour attempt.

Resumed from step 1800 (last verified-good checkpoint) with the eval 
disabled going forward.

____
Depth=20 pretraining -- resumed from step 1800, ran to completion

After the step-2000 eval crashes were fixed (core eval disabled 
permanently), training resumed from step 1800 and ran cleanly the rest 
of the way -- no further crashes. Full 3320 steps completed.

Final result: CORE metric 0.2274, validation bpb 0.757. Sample 
generations showed real, coherent, factually correct sentences 
("capital of France is Paris", "chemical symbol of gold is Au") -- 
a clear, visible improvement over the old depth=12 base model, which 
got stuck in repetition loops on the same kind of prompts. This 
confirmed the core hypothesis behind the whole depth=20 decision: model 
capacity, not training procedure, was the real limitation before.

Base checkpoint saved at:
/workspace/nanochat_checkpoints_d20/base_checkpoints/d20/model_003320.pt

#### SFT re-run on the new depth=20 base model

Confirmed nanochat's setup_optimizer() automatically rescales learning 
rate based on model size -- so the existing SFT script worked correctly 
on the bigger model with no manual math needed.

Reused the best SFT script from the depth=12 experiments (grad_accum=64), 
pointed at the new base model and new result folders.

Run 1 (grad_accum=64): finished cleanly, no crashes. Best checkpoint at 
step 6000, val loss 2.5619 -- clearly better than the best depth=12 
result (2.8877). Output quality also improved: cleaner grammar, better 
topic accuracy, first signs of correctly identifying the narrator's own 
perspective instead of inverting it.

Run 2 (grad_accum=128, doubled again): finished cleanly. Best checkpoint 
at step 36,000, val loss 2.5310 -- smaller improvement than the last 
doubling, consistent with diminishing returns as batch size increases. 
Output quality: best results yet on 2 of 3 test examples, including the 
first checkpoint to fully correctly identify the narrator as the jealous 
one (a persistent error in every prior checkpoint). One example (#25) 
was a clear regression -- garbled, incoherent output. Noted as a real, 
recurring weak point tied to that specific example's structure, not a 
new problem -- worth mentioning honestly in the writeup rather than 
hiding it.

Decision: stopped here. Diminishing returns plus a genuine fix on the 
main qualitative problem (perspective inversion) made this a reasonable 
stopping point rather than chasing further incremental batch-size gains.

Final SFT checkpoint (Phase 3 deliverable):
/workspace/nanochat_checkpoints_d20/sft_d20_v2_best/d20/model_036000.pt

#### Phase 4 (reward model) -- started

Read InstructGPT's reward modeling section. Core concept understood: 
humans give inconsistent absolute scores but reliable pairwise 
judgments (A vs B, which is better). The reward model learns a single 
score per summary such that the DIFFERENCE between two scores, passed 
through a sigmoid function, predicts the human's actual pairwise choice. 
Same "compute error, adjust the model" mechanism as every phase before, 
just a different comparison target -- not absolute correctness, relative 
preference.

Analogy noted: like chess Elo ratings -- no single game gives a number 
directly, but enough win/loss results produce a consistent, comparable 
score for every player.

Next step in progress: loading the actual preference dataset 
(openai/summarize_from_feedback, 'comparisons' config) to inspect one 
real example's structure before writing any reward model code.

____
More notes from this seciton:
#### Reward model concept -- deeper explanation for review

What problem this phase actually solves

SFT teaches the model to imitate ONE example per prompt. It has no idea 
if that example was slightly good, very good, or amazing -- it just 
copies it. This means SFT can't tell the difference between "decent" 
and "excellent" output, only "matches training example" vs "doesn't."

The reward model's job is to fix that gap: give the model a way to 
measure QUALITY on a sliding scale, not just imitate a fixed answer.

Why pairwise comparisons instead of direct scores

Humans are bad at consistent absolute ratings. Ask the same person to 
score a summary today vs next week, the number drifts. But humans are 
good at "which of these two is better" -- that judgment stays stable.

So the training data is built entirely from these simple A-vs-B choices, 
never a raw number.

How binary choices turn into a smooth score (the actual mechanism)

1. The reward model looks at a summary and outputs one raw number -- 
   call it its "score." No inherent meaning yet, just a number.

2. For a pair (summary A, summary B), take scoreA minus scoreB.

3. Pass that difference through a sigmoid function -- squashes any 
   number into a 0-to-1 range, shaped like an S-curve. Big positive 
   difference = probability near 1 (very confident A wins). Big 
   negative = near 0. Near zero difference = uncertain, close to 50/50.

4. Compare that predicted probability against what the human actually 
   picked. If the human picked A, the "correct" probability was 1. 
   Compute how wrong the model's prediction was, adjust the model's 
   weights, same backprop + optimizer mechanic as SFT.

5. Repeat across thousands of pairs. The model is never told "what is 
   an 8/10 summary" -- it just keeps getting nudged toward giving better 
   summaries higher raw scores than worse ones, pair after pair.

Why this produces a real, usable scale (not just pairwise labels)

Because the SAME scoring function is reused across every summary the 
model ever sees, not one function per pair. If A beats B, and B beats C, 
the model is pressured to keep giving A > B > C consistently, even 
though it never saw A and C compared directly. Enough pairs force a 
single, internally consistent ranking to emerge.

Analogy: chess Elo rating. No single match produces a number. But after 
many win/loss results, a smooth rating emerges that lets you compare 
even two players who've never played each other.

How this connects to Phase 5 (PPO), coming next

Once the reward model can score any summary, Phase 5 uses that score as 
the training signal to further improve the SFT model -- generate a 
summary, ask the reward model "how good is this," and adjust the SFT 
model to produce higher-scoring summaries over time. This is the actual 
"RLHF" part -- reward model = the judge, PPO = the process that listens 
to the judge and improves.

Key structural difference from every phase so far

Phase 1-3 all shared one mechanism: predict the next token, check if 
right, adjust. Phase 4 breaks that pattern -- it's not predicting text 
at all. It's outputting a single number and being judged on whether that 
number correctly ranks two things against each other. Different loss 
function, different task shape, but same underlying training loop 
structure (forward pass, compute loss, backward pass, optimizer step).

__________

## September 8 (2hr)

Notes on each step:

Reward data: CarperAI/openai_summarize_comparisons (Parquet mirror of 
OpenAI's dataset, original uses a deprecated loading script). Fields: 
prompt, chosen, rejected -- already pre-sorted by human preference, no 
separate label needed.
Built the reward model. It reuses the same base transformer as SFT, but 
swaps out the final layer -- instead of predicting the next word, it 
outputs one single number (a score) for a whole piece of text.

Had to fix one bug: the model's internal math runs in a lower-precision 
number format (bfloat16), but the new scoring layer defaulted to a 
different format (float32). Fixed by matching the two.

Tested it on a made-up sentence -- got back one number, no errors. Score 
itself means nothing yet since this new layer hasn't been trained -- just 
confirming the plumbing works before training it for real.

First real test of the reward model's loss function. Gave it one human 
preference pair -- it scored the WORSE summary higher than the better one 
(expected, since it hasn't learned anything yet, still random). The loss 
number correctly came out high, meaning the math is working: bad guesses 
produce high loss, which is exactly what training will fix.



____
## September 9 (1.5hr)
#### Finished phase 4, moving to phase 5.
Its conceptually harder so, heres the concept:
The core idea, tied to everything you've built so far: PPO takes your SFT model and improves it further, using your reward model as a "teacher" — but how it does that is fundamentally different from SFT's mechanism. SFT trained on fixed, pre-written examples. PPO has the model generate its own summaries, scores them with the reward model, and adjusts the model to produce higher-scoring summaries over time. The model is learning from its own outputs, not from a fixed dataset.
Four separate models work together in this phase — a new concept, not seen before:
Policy model — your SFT model, the one actually being trained and improved right now
Reference model — a frozen, unchanging copy of that same SFT model, kept purely for comparison
Reward model — what you just finished building, used to score outputs
Value model — a new, small helper model that predicts "how good is this partial summary likely to end up," used to make training more stable

### Reading PPO paper (https://arxiv.org/pdf/1707.06347)
Abstract/intro:
Lay out the dillima in Reinforcement learning. Algorithms right now are too fragile and wasteful with data, and too complex. Pro solves this by creating a simple safety guard clipping that lets the AI safely reuse recorded experiences multiple times without taking wrong steps.
PPO: You put a physical stopper on the steering wheel that limits turns to 10 degrees at a time. Because you know you can't accidentally flip the car, you can safely practice driving on the same stretch of road multiple times in a row.

Section 2: Lays out the math and explains why the previous methods either broke or where to complex.
2.1
Shows the standard approach to policy optimization uses gradient ascent to make high reward actions more probable. 
2.2 (TRPO)
Trust Region Policy Optimization introduces a strict safety boundary based on probability ratios. Underlying Concept: Maximize reward while strictly constraining the KL Divergence (the distance between the old and new action distributions) to be below a hard threshold: The flaw is it creates a hard constraint makes it solve multiple operations from different points, instead of all at once, being inefficient. 

Section 3
Old RL is bad because it makes big strategy updates to the models data which can send it to be completely off track. Solutions like TRPO helped prevent the catastrophic ones using the heavy constraints that were slow to compute. PPO solves this by placing a simple numerical cap on how much the ai can alter it decisoin capabilities in one step. By using the rule, PPo prevents the model from overheating to sudden rewards or penalties, enhancing the improving of RM.

Section 4/5
Adding the clipping of the PPOs primary mechinism, section 4 explores the other approach that dynamically increases/decreases a penalty whenever the ai drifts to far, or moves to slowly from the previous state. On top of this, section 5 combines these components into a framework that computes a single unified loss function. The combined goal balances 3 goals at once: Improving desicoin making through clipped rewards, training a critic network to estimate future outcomes, and adding exploration bonus to prevent the model from getting stuck in repetative loops. These bundled, allow an agent to execute multiple training passes on a single batch of collected data using “standard gradient ascent”.

Sections 6 and 7 evaluate PPO across a broad range of continuous robotics simulations and 2D Atari video games, demonstrating that simple objective clipping consistently matches or outperforms far more complex algorithms like TRPO and ACER.

#### Gradient Ascent =
Imagine a model's performance as a topographic map where elevation represents performance or reward. Gradient ascent calculates the slope (gradient) at the model's current parameter configuration and moves those parameters step-by-step uphill toward higher reward peaks.


#### Phase 5 so far:

So far in Phase 5, we've just set up the pieces PPO needs before any actual training happens. We loaded your finished SFT model twice — once as the "policy" (the version that will actually get trained and improved) and once as a frozen "reference" copy that never changes, which exists purely so training can check how far the policy has drifted from where it started. We also loaded your trained reward model from Phase 4 (fixing a real gap along the way — it turned out we'd never actually saved its trained weights to disk the first time, so we reran that training and saved it properly this time) and connected it in as well. Right now we have three of the four required models loaded and confirmed working together; the last piece, a small value model that predicts how good a summary is likely to turn out while it's still being generated, hasn't been built yet.


____
## September 10 (2hr)

next is building the value model, the fourth and final piece needed before PPO's actual training loop
The value model's job, in plain terms: while the policy model is generating a summary token by token, the value model looks at what's been generated so far and predicts "given this partial summary, how good do I think the final reward score will end up being." This running prediction is what makes PPO's updates stable — instead of only learning from the final reward at the very end, the model gets a sense of progress at every step along the way.

- All 4 models ran successful

Next: the actual PPO training loop itself — the most complex piece of this entire project. Given its complexity, we'll build it in stages, same discipline as SFT

#### notes.md digest
Phase 5 setup complete: all four PPO models (policy, frozen reference, 
frozen trained reward model, fresh value model) load together in memory 
with room to spare (10GB used of 24GB). Confirmed the policy model can 
generate real summaries within this same setup -- this is the actual 
text PPO will score and learn from, not fixed dataset examples like SFT used.

Confirmed the reward model can score the policy's own live-generated 
text, not just fixed dataset examples. This is the actual signal PPO 
will use to improve the model -- generate a summary, score it, then 
adjust the model to produce higher-scoring summaries over time.

KL divergance, the way to tell how far the model has drifted from where it started 
Generated summary:  I think it's a good thing to open a secret communication that might be able to show her that she's hurt so and wants to change the rut we both live in (which is really starting to open), but she wants us back together.
Reward score for generated summary: 70.0
KL divergence (policy vs reference): 0.0

#### Here's how the four pieces combine into PPO's actual training signal

The goal: make the policy generate summaries that score higher, without letting it change too recklessly in one step.
Advantage: reward score minus what the value model expected. Positive means "better than expected", do more of this, but  negetive is worse than expected, do less of that.
The safety trick (what "PPO" actually means): instead of just pushing hard toward high-advantage outputs, it compares the model's current confidence in a token to its confidence just before this update, and caps how much that ratio is allowed to shift.
The drift penalty: before any of this, we subtract a penalty based on how far the policy's token probabilities have drifted from the frozen reference model. Score well, but drift too far from where you started, and the effective reward gets reduced. This is the actual mechanism that stops reward hacking (no cheating), and is lead by KL divergence

generate → score → penalize for drifting too far → compare to expectations → nudge gently, capped, toward what worked → also teach the value model to predict better next time.

___
## September 11 (3hr)

#### Now running first real PPO train (100 steps, save every 20)
cd /workspace/VerdictGPT
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache
export HF_HUB_DISABLE_XET=1
export NANOCHAT_BASE_DIR=/workspace/nanochat_checkpoints_d20
export PYTHONPATH=/workspace/nanochat:$PYTHONPATH
python3 ppo_train_full.py


Inside screen session, reatach with: screen -r ppo_final

After that run, it seems that the reward value was being weird, varying from -70 to 60, and no steady or clear ascent trend. also some consecutive rollouts show the same rewards, which is no good, means it got stuck in a loop, generating duplicates for a bit.
But before making up offsetting predictions, lets test it, and compare runs before and after the training.

Failed.
=== Example 0 ===
Generated:  Where the fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck

=== Example 5 ===
Generated:  Why the high interest loan payment and opt for a minimal downpayment? if a lower interest loan payment is beneficial to the business.  What would you advise in my shoes towards a company that has had no problems? 

=== Example 10 ===
Generated:  Where the fuck do I put this? It's hard to talk to HR if there's something wrong or what I do and what I should do about it.

(.venv) root@b562fa4d4dc6:/workspace/VerdictGPT# 

This is a collapse, the model found a shortcut phrase (fuck) as it tends to score well on the reward model.

Conduct test after small change, in normal terminal, no screen, changed number of rollouts (only 10)

Now 6different sections of runs, train, full, rollout, test, show_result, and reward_sanity
Found out its not a broken reward model from runing sanity check rollout, in the check, the garbage text scored lower than the normal summary. So its actuially the ppo training loops fault

#### Hopefullt:
real, confirmed root cause. Every 4 consecutive indices show the exact same post. This is the actual bug, and it explains everything: not a PPO instability problem, not a reward model problem, not a missing-clipping problem. The dataset itself has each unique post repeated 4 times in a row — almost certainly because CarperAI/openai_summarize_comparisons stores multiple comparison pairs per post (a common structure: one post, several different human-judged summary pairs), and consecutive rows share the same post with different summary pairs, not different posts entirely.

### Failed

PPO was implemented with all four required models (policy, frozen reference, trained reward model, value model) working correctly together. During testing, a real data bug was found and fixed: the dataset was accidentally serving the same post four times in a row, which had been causing the model to generate identical output regardless of input. Once fixed, PPO started training on genuinely varied data — but revealed a separate, real problem: the model learned to chase the reward model by injecting aggressive language ("Why the fuck...") into nearly every summary, regardless of the actual post content. This is a well-known, documented PPO failure mode called reward hacking, not a bug in the code — it happens because the reward model, trained on real but imperfect human preference data, imperfectly rewards this kind of language.

Decision: stop here. The SFT model remains the best-quality, most reliable output from this project and will be used as the actual deliverable. This PPO result is kept and documented as a real, working implementation that surfaced a genuine, well-known limitation of the technique — not a failure to get something running, but an honest demonstration of why RLHF is hard to get right in practice.

__________
# DONE!!!

I learned so much in this project and dont regret anything about it. it is. acool thing to publish, and im excited to show other.

#### Project closing

VerdictGPT is done. Built a full RLHF pipeline from scratch -- pretraining, 
SFT, a reward model, and PPO -- on a single rented GPU, without using any 
library that hides the actual algorithms.

Real numbers: 897M parameter base model (after an earlier 286M attempt hit 
a real capacity ceiling), SFT val loss down to 2.53 after a real batch-size 
search, reward model at ~57% preference accuracy, PPO trained on top of all 
of it with a working KL penalty and value model.

Real problems hit and actually fixed, not glossed over: repeated disk 
quota crashes, a duplicate-checkpoint bug that silently ate local storage, 
a corrupted resume file, and a dataset indexing bug that looked exactly 
like a training collapse until traced back to its real, boring cause. 
Every long script ended up with proper checkpoint saving and resume 
support -- none of that existed in the first draft of any of them.

Final result is an honest one: PPO didn't produce a clean, polished model. 
It inherited a real, visible bias from the base model's pretraining data 
(profanity showing up regardless of topic) and never fully escaped it. 
That's not a hidden flaw -- it's documented, demonstrated live, and 
explained in the writeup as a real example of why RLHF is hard to get 
right, not just hard to set up.

Everything is pushed to GitHub, organized by phase, with a working live 
demo anyone can run and test with their own input. Notes, writeup, and 
code all exist and all point to the same honest story.
_________



### Final push + how to run the demo

Git commit got stuck for a while -- turned out .gitignore never excluded 
.venv/, so `git add .` was trying to stage the entire virtual environment 
(thousands of files). Fixed .gitignore, killed the stuck process, redid 
the commit -- worked in seconds after that. Pushed everything to GitHub: 
code, WRITEUP.md, README.md, organized into phase folders. Checkpoints and 
data stay off GitHub on purpose (too large, correctly excluded).

Built a live demo script (demo/live_demo.py) that loads the final PPO 
model once and lets you paste in any real paragraph to get a live 
summary -- better than showing a canned example, since it proves the 
model actually works on the spot.

## How to run the demo for someone

1. Confirm the RunPod pod is running (check dashboard, start if stopped)
2. SSH in with the current connection string from the dashboard
3. Run:
   cd /workspace/VerdictGPT
   source .venv/bin/activate
   export HF_HOME=/workspace/hf_cache
   export HF_HUB_DISABLE_XET=1
   export NANOCHAT_BASE_DIR=/workspace/nanochat_checkpoints_d20
   export PYTHONPATH=/workspace/nanochat:$PYTHONPATH
   python3 demo/live_demo.py
4. Wait for "Ready."
5. Paste any paragraph when prompted, see the summary print live
6. Type "quit" to exit

#### Three example paragraphs to have ready

1. Workplace/pay disagreement:
"I've been working at my company for about three years now and I recently 
found out that a coworker who started six months after me is making 
significantly more money than I am, despite us doing basically the same 
job. I brought this up to my manager and he said budgets are tight right 
now and he can't promise anything, but he'll 'look into it.' That was 
over a month ago and I haven't heard anything back. I really like my job 
and my team, but I'm starting to feel like I'm being taken advantage of. 
Should I bring it up again, start looking at other jobs, or just wait it 
out?"

2. Roommate conflict:
"My roommate and I split rent and utilities evenly, but she's been having 
her boyfriend stay over almost every night for the past two months, and 
our water and electric bills have gone up noticeably. I don't want to be 
petty about it, but it feels unfair that I'm paying half the cost of 
someone who doesn't even live here. I haven't said anything yet because I 
don't want to start a fight, but it's been bothering me more each month. 
How do I bring this up without it turning into a huge argument?"

3. Family/inheritance disagreement:
"My grandmother passed away last year and left her house to be split 
evenly between me and my two siblings. My older brother has been living 
in the house rent-free for the past eight months while we sort out the 
sale, and he keeps pushing back the timeline every time we bring it up. 
My sister and I could really use our share of the money, but every 
conversation turns into an argument about how we're being 'insensitive' 
about grandma's memory. I don't know if we should get a lawyer involved 
or just keep waiting."

