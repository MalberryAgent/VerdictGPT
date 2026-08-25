# Notes for project creation

### Contains:
- Changes made
- Learning curves
- Progress timeline
- Challenges

## Notes


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


 # August 25 (1hr)

 
