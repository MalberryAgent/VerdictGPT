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
- Name: torch
Version: 2.8.0+cu128
- Name: transformers
Version: 5.15.1
- Name: datasets
Version: 5.0.1
- Name: wandb
Version: 0.28.2

## August 23 - (40m)
Watching first vid tutorial!
### Let’s build ChatGPT: From scratch, in code, , spelled out. - Andrej Karpathy

ChatGPT is transformer architecture as written in a research paper in 2017 “Attention is all you need” and implemented in GPTs models.
Tutorial: Build a pre trained GPT from scratch and using package to inspect and adjust internal mechanics at the code level, (CHECK OUT NANOGPT REPO)
Pre training is training a neural network on massive web datasets to predict the next token, producing a document completer, not assistant, then fine tune base model with QA written by human, then RLHF (Reinforcement Learning from Human Feedback) to align model, and reward model.
Begins with empty file, define  transformer, then train on Shakespeare dataset, and generate Shakespeare infinitely.
words of Shakespeare text as set, and list on top to create order and meaning, and then it sorts.
Strategy to tokenize, take raw string of text to integers to be vocab for possible  elements, characters -> Integers
Hi there = [41, 56, 23, 67, 23, 67, 46, 15] You can encode and decode, this is one of many possible methods. This is character level tokenizer.
Tokenize library of Shakespeare, all the text, is a very large sequence of integers.
To improve efficiency and future proof, save 90% of library to train, then remaining 10% to validate.
Train transformer
Train by chunks with maximum length, not all at once because then to much compute.
Think of prediction like this
Hi there = [18, 47, 56, 87, 13, 45]
In context of 18, 47 likely comes next. In context of 18, 47, the next is likely 56. In context of 18, 47, 56, then 87 is likely next.
5 Examples hidden in a chunk of 6
A data loader prevents memory crashes by chopping a giant text library into small snippets called chunks. It then groups these chunks into batches so the computer can learn from multiple snippets at the exact same time, handing and structuring for the ai training.

##### Building simplest AI possible - Bigram Model.
Tries to predict next letter only looking at the one before, like seeing Q, and guessing U. Measuring loss track progress, its like a mistake score, the Lowe the number, the better its guessing. SO an untrained model would have a high loss and it would be gibberish. This is the structure to set up. AI can guess the next letter, measure how wrong then generate text.
Next, training the bigram model by running it through thousands of training loops to teach it using a tool called an optimizer. The code grabs text, measure the loss, calculates what number caused error, then nudges in the right direction slightly, gibberish turns into English like word patterns.
