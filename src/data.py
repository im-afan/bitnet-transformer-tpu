import torch
from transformers import AutoTokenizer

class Tokenize(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if(self.tokenizer.pad_token is None):
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch):
        text = batch['text']

        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )

        return {'text': tokenized['input_ids']}



