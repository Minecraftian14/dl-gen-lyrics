import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from dl_trainer import Trainer


class ConditionalDataset(data.Dataset):
    def __init__(self, midnight: 'Midnight'):
        self.midnight = midnight

    def __len__(self):
        return len(self.midnight.ds_data)

    def __getitem__(self, index):
        midnight = self.midnight
        sample = midnight.ds_data.iloc[index]
        lyrics = sample['lyrics']
        tag = sample['tag']

        tokens = midnight.tokenize_text(lyrics)
        context = midnight.tokenize_text(" ".join(midnight.get_context_words(lyrics)))
        genre = midnight.genre_to_id[tag]
        return tokens, context, genre

    def collate_fn(self, batch):
        tokens, context, genres = zip(*batch)

        input_ids = [torch.tensor(t[:-1], dtype=torch.long) for t in tokens]
        target_ids = [torch.tensor(t[1:], dtype=torch.long) for t in tokens]
        context_ids = [torch.tensor(c, dtype=torch.long) for c in context]

        lengths = torch.tensor([len(t) for t in input_ids], dtype=torch.long)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)
        context_ids = pad_sequence(context_ids, batch_first=True, padding_value=0)

        genres = torch.tensor(genres, dtype=torch.long)

        return (input_ids, context_ids, lengths, genres), target_ids

    def criteria_step_fn(self, criterion, logits, target_ids):
        B, T, V = logits.shape
        return criterion(logits.view(B * T, V), target_ids.view(B * T))


class ConditionalLSTMLM(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_size,
            num_layers,
            num_genres,
            genre_emb_dim=16,
            word2vec_weights=None,
            word2vec_frozen=True,
            pad_idx=0,
            dropout=0.2,
    ):
        super().__init__()

        # Word embeddings
        # Basically, we train a word2vec beforehand, and then pass it on here
        if word2vec_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                word2vec_weights, freeze=word2vec_frozen, padding_idx=pad_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Genre embedding
        # A separate embedder for genre, learnt in here itself
        self.genre_embedding = nn.Embedding(num_genres, genre_emb_dim)

        # Conditioning projection → hidden state
        # These layers basically help match the emb dims from the embeddings to the internal state of the lstm
        # Note, embedding_dim is for context words, and genre_emb_dim is for genre word, which are concatenated
        self.condition_proj_h = nn.Linear(embedding_dim + genre_emb_dim, hidden_size)
        self.condition_proj_c = nn.Linear(embedding_dim + genre_emb_dim, hidden_size)

        # LSTM decoder
        # The heart of our algorithm here
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, context_ids, lengths, genre_ids, hidden=None):
        """
        :param input_ids:   (B, T) for the lyric data
        :param context_ids: (B, C) 1-5 words of context
        :param lengths:     (B,) actual sequence lengths (before padding)
        :param genre_ids:   (B,) genre of the lyric
        :param hidden:
        :return:
        """

        # Firstly, prepare the data (get the embeds for lstm)

        # Embeds for the input
        emb = self.embedding(input_ids)  # (B, T, E)
        # Embeds for the context
        context_emb = self.embedding(context_ids)  # (B, C, E)
        # For now... Let's just mean it... IDK what will happen.
        context_vec = context_emb.mean(dim=1)  # (B, E)
        # Embeds for the genre
        genre_vec = self.genre_embedding(genre_ids)  # (B, G)

        # Then apply conditioning, where we are simply concatenating and reshaping using linear transform
        cond = torch.cat([context_vec, genre_vec], dim=-1)  # (B, E+G)
        h0 = torch.tanh(self.condition_proj_h(cond))  # (B, H)
        c0 = torch.tanh(self.condition_proj_c(cond))  # (B, H)

        # Do the lstm thing (Expand to (num_layers, B, H) which is what the LSTM expects)
        h0 = h0.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)

        # Apply packing to avoid computations on pads
        packed_emb = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Call on the LSTM
        packed_outputs, _ = self.lstm(packed_emb, (h0, c0))
        # Unpack the outputs
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)  # (B, T, H)

        return self.output_layer(outputs)  # (B, T, V)

    def _optimizer(self, parameters):
        return optim.AdamW(parameters, lr=0.001)

    def prepare_train(self, ds_data: ConditionalDataset):
        self.dataloader = data.DataLoader(
            ds_data,
            batch_size=16,
            shuffle=True,
            collate_fn=ds_data.collate_fn,
        )
        self.trainer = Trainer(
            model=self,
            train_dataloader=self.dataloader,
            criterion=nn.CrossEntropyLoss(ignore_index=0),
            optimizer=self._optimizer,
            epochs=1,
            device='cpu',
            record_per_batch_training_loss=True,
            model_criteria_step=ds_data.criteria_step_fn,
        )

    def train_model(self):
        self.trainer.train()
