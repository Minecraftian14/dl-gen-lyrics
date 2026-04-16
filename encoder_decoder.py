import re
import csv
import sys
import os
import torch
from torch import nn
from torch.utils.data import IterableDataset
import sentencepiece as spm

# Increase CSV field size limit for long lyrics
csv.field_size_limit(sys.maxsize)

# --- 1. MODEL ARCHITECTURE ---
class EncoderDecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()
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
    def __init__(self, vocabulary, data_generator, seq_len=15, limit=None):
        self.vocabulary = vocabulary
        self.data_generator = data_generator
        self.seq_len = seq_len
        self.limit = limit

    def __iter__(self):
        for genre_token, song_text in self.data_generator(limit=self.limit):
            encoded_ann = [self.vocabulary.get_id(genre_token)]
            encoded_song = [self.vocabulary.get_id("<SOS>")] + \
                           self.vocabulary.encode(song_text) + \
                           [self.vocabulary.get_id("<EOS>")]

            if len(encoded_song) <= self.seq_len: continue
            
            for i in range(len(encoded_song) - self.seq_len):
                window_x = encoded_song[i : i + self.seq_len]
                target_y = encoded_song[i + self.seq_len]
                yield torch.tensor(encoded_ann), torch.tensor(window_x), torch.tensor(target_y)

def collate_seq2seq(batch):
    anns, windows_x, ys = zip(*batch)
    
    max_ann_len = max(len(a) for a in anns)
    padded_anns = torch.zeros(len(anns), max_ann_len, dtype=torch.long)
    for i, a in enumerate(anns): padded_anns[i, :len(a)] = a
        
    windows_x = torch.stack(windows_x)
    ys = torch.stack(ys)
    
    return padded_anns, windows_x, ys

# --- 4. TEXT PROCESSING UTILS ---
def simplify_lyrics(lyrics: str):
    lyrics = re.sub(r" +", " ", lyrics)
    lyrics = re.sub(r"[^\w\n., ]", "", lyrics)
    return lyrics.lower().strip()