"""Main training (fine-tuning) script."""

import argparse

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# Download configuration from huggingface.co and cache.
from transformers import Trainer
from transformers import TrainingArguments

import data
import generate

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(
    description="Arguments for training and evaluation")
parser.add_argument('--n_epochs', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--save_steps', type=int, default=500)
args = parser.parse_args()

# Generate initial output
generate.generate('gen_init', model_dir=None)

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# Freeze some model layers to stabilize fine-tuning
for parameter in model.parameters():
    parameter.requires_grad = False

for i, m in enumerate(model.transformer.h):
    # Only un-freeze the last n transformer blocks
    if i >= 6:
        for parameter in m.parameters():
            parameter.requires_grad = True

for parameter in model.transformer.ln_f.parameters():
    parameter.requires_grad = True

for parameter in model.lm_head.parameters():
    parameter.requires_grad = True

training_args = TrainingArguments("test_trainer")
training_args.num_train_epochs = args.n_epochs
training_args.learning_rate = args.lr
training_args.batch_size = args.batch_size
training_args.save_steps = args.save_steps

train_dataset = data.get_dataset()

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=train_dataset)
trainer.train()
model.save_pretrained(save_directory='models/gpt2_homer')
generate.generate('gen_final', model_dir='models/gpt2_homer')
