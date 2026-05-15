import torch
from transformers import AutoTokenizer

class Tokenize(object):
    def __init__(self, max_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained("georgeyw/TinyStories-tokenizer-10k")
        if(self.tokenizer.pad_token is None):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def __call__(self, batch):
        text = batch['text']

        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        return {'text': tokenized['input_ids']}



