from collections import Counter
from typing import Iterable

import spacy
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class Vocab:
    def __init__(self, tokens: Iterable[list[str]], min_freq: int = 2):
        specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
        counter = Counter(token for sentence in tokens for token in sentence)
        self.itos = list(specials)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

        for token, freq in counter.most_common():
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

        self.unk_idx = self.stoi["<unk>"]
        self.pad_idx = self.stoi["<pad>"]
        self.sos_idx = self.stoi["<sos>"]
        self.eos_idx = self.stoi["<eos>"]

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.unk_idx)

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.sos_idx] + [self[token] for token in tokens] + [self.eos_idx]


class Multi30kDataset(Dataset):
    _raw_cache = None
    _vocab_cache = None

    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.tokenizer_de = self._load_tokenizer("de")
        self.tokenizer_en = self._load_tokenizer("en")

        if Multi30kDataset._raw_cache is None:
            Multi30kDataset._raw_cache = load_dataset("bentrevett/multi30k")

        split_name = "validation" if split in {"valid", "val"} else split
        self.raw_data = Multi30kDataset._raw_cache[split_name]

        if Multi30kDataset._vocab_cache is None:
            Multi30kDataset._vocab_cache = self.build_vocab()

        self.src_vocab, self.tgt_vocab = Multi30kDataset._vocab_cache
        self.data = self.process_data()

    @staticmethod
    def _load_tokenizer(lang: str):
        model_name = {"de": "de_core_news_sm", "en": "en_core_web_sm"}[lang]
        try:
            return spacy.load(model_name)
        except OSError:
            return spacy.blank(lang)

    def _get_text_pair(self, example) -> tuple[str, str]:
        if "de" in example and "en" in example:
            return example["de"], example["en"]
        if "translation" in example:
            return example["translation"]["de"], example["translation"]["en"]
        raise KeyError("Expected Multi30k example to contain de/en text fields")

    def _tokenize_de(self, text: str) -> list[str]:
        return [token.lower_ for token in self.tokenizer_de(text)]

    def _tokenize_en(self, text: str) -> list[str]:
        return [token.lower_ for token in self.tokenizer_en(text)]

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # Vocabularies are built from the training split only to avoid leakage.
        train_data = Multi30kDataset._raw_cache["train"]
        src_tokens = []
        tgt_tokens = []
        for example in train_data:
            de_text, en_text = self._get_text_pair(example)
            src_tokens.append(self._tokenize_de(de_text))
            tgt_tokens.append(self._tokenize_en(en_text))
        return Vocab(src_tokens), Vocab(tgt_tokens)

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        processed = []
        for example in self.raw_data:
            de_text, en_text = self._get_text_pair(example)
            src = self.src_vocab.encode(self._tokenize_de(de_text))
            tgt = self.tgt_vocab.encode(self._tokenize_en(en_text))
            processed.append((torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)))
        return processed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def collate_batch(batch, pad_idx: int = 1):
    src_batch, tgt_batch = zip(*batch)
    # Pad within each batch after tokenization so the model sees rectangular tensors.
    src = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src, tgt
