# Notes for project creation

### Contains:
- Changes made
- Learning curves
- Progress timeline
- Challenges

## Notes


___________________________________________


## August 22 - (3hr)
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





 # August 25 (3hr)

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

