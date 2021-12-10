"""Data loading, preprocessing, and dataset object creation."""

import torch.utils.data as data
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

class TokenizedSentensesDataset(data.Dataset):
    def __init__(self, tok_sents):
        self.tok_sents = tok_sents

    def __len__(self):
        return len(self.tok_sents['input_ids'])

    def __getitem__(self, idx):
        item = {
            'input_ids': self.tok_sents['input_ids'][idx],
            'attention_mask': self.tok_sents['attention_mask'][idx],
            'labels': self.tok_sents['input_ids'][idx]
        }
        return item

def get_tok_sents():
    batch_sentences = open('iliad_sents.txt').read().splitlines()
    batch_sentences += open('odyssey_sents.txt').read().splitlines()
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_sentences = tokenizer(batch_sentences,
                                    return_tensors='pt',
                                    padding="max_length",
                                    truncation=True)
    return tokenized_sentences

def get_dataset():
    tok_sents = get_tok_sents()
    dataset = TokenizedSentensesDataset(tok_sents)
    return dataset