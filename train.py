"""Main training (fine-tuning) script."""

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM
# Download configuration from huggingface.co and cache.
from transformers import Trainer
from transformers import TrainingArguments

import data
import generate

parser = argparse.ArgumentParser(
    description="Arguments for training and evaluation")
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--save_steps', type=int, default=500)
args = parser.parse_args()

# Generate initial output
# generate.generate('gen_init.txt', model_dir=None)

config = AutoConfig.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_config(config)

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
generate.generate('gen_final.txt', model_dir='models/gpt2_homer')
