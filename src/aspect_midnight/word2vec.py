import random, tqdm

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from dl_trainer import Trainer

SKIPGRAM_N_WORDS = 4
MAX_SEQUENCE_LENGTH = 256


class ArrayToDatasetForW2V(data.Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        return self.array[idx]


class Word2Vec_SkipGram(nn.Module):

    def __init__(
            self,
            # Text Helpers
            text_to_ids,
            # Word2Vec Parameters
            vocab_size: int, d_embeds: int = 300, max_norm: float = 1.0,

    ):
        super(Word2Vec_SkipGram, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_embeds,
            max_norm=max_norm,
        )
        self.linear = nn.Linear(
            in_features=d_embeds,
            out_features=vocab_size,
        )
        self.text_to_ids = text_to_ids

    def forward(self, x):
        x = self.embeddings(x)
        x = self.linear(x)
        return x

    def collate_fn(self, batch):
        assert isinstance(batch, list)
        assert isinstance(batch[0], str)

        batch_input, batch_output = [], []
        for text in batch:
            text_tokens_ids = self.text_to_ids(text)

            if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
                continue

            if MAX_SEQUENCE_LENGTH and len(text_tokens_ids) > MAX_SEQUENCE_LENGTH:
                sid = random.randint(0, len(text_tokens_ids) - MAX_SEQUENCE_LENGTH)
                text_tokens_ids = text_tokens_ids[sid: sid + MAX_SEQUENCE_LENGTH]

            for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
                token_id_sequence = text_tokens_ids[idx: (idx + SKIPGRAM_N_WORDS * 2 + 1)]
                input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
                outputs = token_id_sequence

                for output in outputs:
                    batch_input.append(input_)
                    batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return (batch_input,), batch_output

    def _optimizer(self, parameters):
        return optim.AdamW(parameters, lr=0.025)

    def prepare_train(self, ds_data: data.Dataset):
        self.dataloader = data.DataLoader(
            ds_data,
            batch_size=16,  # 96
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        self.trainer = Trainer(
            model=self,
            train_dataloader=self.dataloader,
            criterion=nn.CrossEntropyLoss(),
            optimizer=self._optimizer,
            epochs=1,
            device='cpu',
            record_per_batch_training_loss=True,
        )

    def train_model(self):
        self.trainer.train()
