import re
import csv
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import IterableDataset, DataLoader
import sentencepiece as spm

from dl_trainer import Trainer
from generator_core import Solution

# Increase CSV field size limit for long lyrics
csv.field_size_limit(sys.maxsize)

# --- 1. MODEL ARCHITECTURE ---
class EncoderDecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, embeddings_weight=None):
        super().__init__()

        if embeddings_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings_weight, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, annotations_x, decoder_x):
        enc_embedded = self.embedding(annotations_x)
        _, (hidden, cell) = self.encoder_lstm(enc_embedded)
        
        dec_embedded = self.embedding(decoder_x)
        outputs, _ = self.decoder_lstm(dec_embedded, (hidden, cell))
        
        last_hidden = outputs[:, -1, :] 
        logits = self.fc(last_hidden)
        
        return logits

    def _optimizer(self, parameters):
        return optim.AdamW(parameters, lr=0.001)

    def prepare_train(self, ds_data: SlidingWindowDataset):
        self.dataloader = DataLoader(
            ds_data,
            batch_size=256,
            collate_fn=collate_seq2seq,
        )
        self.trainer = Trainer(
            model=self,
            train_dataloader=self.dataloader,
            criterion=nn.CrossEntropyLoss(ignore_index=0),
            optimizer=self._optimizer,
            epochs=1,
            device='cpu',
            record_per_batch_training_loss=True,
        )

    def train_model(self):
        self.trainer.train()


# --- 2. VOCABULARY MANAGER ---
class SentencePieceVocab:
    def __init__(self, vocab_size=15000, model_prefix="lyrics_spm"):
        self.vocab_size = vocab_size
        self.sp_model = spm.SentencePieceProcessor()
        self.model_prefix = model_prefix
        self.genre_tokens = ["<pop>", "<rock>", "<rb>", "<misc>", "<country>", "<rap>"]

    def load(self, model_path):
        """Loads an already trained SentencePiece model."""
        self.sp_model.load(model_path)

    def encode(self, text):
        return self.sp_model.encode_as_ids(text)

    def decode(self, ids):
        return self.sp_model.decode_ids(ids)

    def get_id(self, token):
        if token == "<PAD>": return self.sp_model.pad_id()
        if token == "<UNK>": return self.sp_model.unk_id()
        if token == "<SOS>": return self.sp_model.bos_id()
        if token == "<EOS>": return self.sp_model.eos_id()
        return self.sp_model.piece_to_id(token)

# --- 3. DATASET & PACKING ---
class SlidingWindowDataset(IterableDataset):
    def __init__(self, red: Solution, seq_len=15, limit=None):
        self.red = red
        self.seq_len = seq_len
        self.limit = limit

    def __iter__(self):
        dataset = self.red.ds_data.itertuples()
        iterator = enumerate(dataset) if self.limit is None else zip(range(self.limit), dataset)

        for _, sample in iterator:
            encoded_ann = self.red.tokenize_text(sample.tag)
            encoded_ctx = self.red.tokenize_text(" ".join(self.red.get_context_words(sample.lyrics)))
            encoded_song = self.red.tokenize_text(sample.lyrics)

            encoded_ann.extend(encoded_ctx)

            if len(encoded_song) <= self.seq_len: continue
            
            for i in range(len(encoded_song) - self.seq_len):
                window_x = encoded_song[i : i + self.seq_len]
                target_y = encoded_song[i + self.seq_len]
                yield torch.tensor(encoded_ann), torch.tensor(window_x), torch.tensor(target_y)


class SlidingWindowDatasetTruncated(IterableDataset):
    def __init__(self, red: Solution, seq_len=15, limit=None):
        self.red = red
        self.seq_len = seq_len
        self.limit = limit

    def __len__(self):
        length = len(self.red.ds_data) if self.limit is None else self.limit
        return length * 100

    def __iter__(self):
        dataset = self.red.ds_data.itertuples()
        iterator = enumerate(dataset) if self.limit is None else zip(range(self.limit), dataset)

        for _, sample in iterator:
            encoded_ann = self.red.tokenize_text(sample.tag)
            encoded_ctx = self.red.tokenize_text(" ".join(self.red.get_context_words(sample.lyrics)))
            encoded_song = self.red.tokenize_text(sample.lyrics)

            encoded_ann.extend(encoded_ctx)

            if len(encoded_song) <= self.seq_len: continue

            indices = np.random.randint(1, len(encoded_song) - self.seq_len - 1, 98).tolist()
            indices = [0] + indices + [len(encoded_song) - self.seq_len - 1]

            for i in indices:
                window_x = encoded_song[i: i + self.seq_len]
                target_y = encoded_song[i + self.seq_len]
                yield torch.tensor(encoded_ann), torch.tensor(window_x), torch.tensor(target_y)



def collate_seq2seq(batch):
    anns, windows_x, ys = zip(*batch)
    
    max_ann_len = max(len(a) for a in anns)
    padded_anns = torch.zeros(len(anns), max_ann_len, dtype=torch.long)
    for i, a in enumerate(anns): padded_anns[i, :len(a)] = a
        
    windows_x = torch.stack(windows_x)
    ys = torch.stack(ys)

    return (padded_anns, windows_x), ys

# --- 4. TEXT PROCESSING UTILS ---
def simplify_lyrics(lyrics: str):
    lyrics = re.sub(r" +", " ", lyrics)
    lyrics = re.sub(r"[^\w\n., ]", "", lyrics)
    return lyrics.lower().strip()