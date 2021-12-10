"""Main training (fine-tuning) script."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM
# Download configuration from huggingface.co and cache.
from transformers import Trainer
from transformers import TrainingArguments

import data

CKPT_INT = 100
USE_CUDA = torch.cuda.is_available()

config = AutoConfig.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_config(config)

training_args = TrainingArguments("test_trainer")
training_args.num_train_epochs = 100

train_dataset = data.get_dataset()
trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=train_dataset)
trainer.train()
model.save_pretrained(save_directory='models/gpt2_homer')