import sys

import sentencepiece as spm
import torch.nn.functional as F
from torch import optim
from torch.utils.data import IterableDataset, DataLoader

from dl_trainer import Trainer
from generator_core import *
from generator_core import Solution
from models import Midnight

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

    def forward(self, annotations_x, decoder_x, get_outputs=False):
        enc_embedded = self.embedding(annotations_x)
        _, (hidden, cell) = self.encoder_lstm(enc_embedded)

        dec_embedded = self.embedding(decoder_x)
        outputs, _ = self.decoder_lstm(dec_embedded, (hidden, cell))

        if get_outputs: return self.fc(outputs)

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
            epochs=15,
            device='cuda' if torch.cuda.is_available() else 'cpu',
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
                window_x = encoded_song[i: i + self.seq_len]
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
            encoded_ann = self.red.tokenize_text(sample.tag + " " + " ".join(self.red.get_context_words(sample.lyrics)))
            encoded_song = self.red.tokenize_text(sample.lyrics)

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


sure_fire_keywords = [
    "chorus", "verse", "stanza", "interlude",
]
two_fire_keywords = [
    "intro", "outro",
    "bridge",
    "hook",
]
other_words = [
    "music",
    "pre-", "post-",
]
structure_tokens = [
    "[", "]", "(", ")", "<", ">", "{", "}",
    ":",
]


class Red(Solution):

    def __init__(self, ds_data: pd.DataFrame):
        """
        Initialize all mid-state helpers
        """

        # For accessing shared tokenizer and embeddings
        self.midnight = Midnight(ds_data, skip_model_loading=True)

        self.ds_data: pd.DataFrame = self.midnight.ds_data
        self.custom_tokens = self.midnight.custom_tokens
        self.genre_to_id = self.midnight.genre_to_id
        self.id_to_genre = self.midnight.id_to_genre
        self.tfidf = self.midnight.tfidf
        self.feature_names = self.midnight.feature_names
        self.vocabulary = self.midnight.vocabulary
        self.embedder = self.midnight.embedder
        self.language_model = self._prepare_language_model()

    @cached()
    def _prepare_language_model(self):
        edm = EncoderDecoderLSTM(
            vocab_size=self.vocabulary.vocab_size(),
            embed_dim=512,
            hidden_dim=512,
            num_layers=2,
            embeddings_weight=self.embedder.embeddings.weight,
        )
        edm.prepare_train(SlidingWindowDatasetTruncated(self))
        return edm

    def get_data_size(self) -> int:
        return self.midnight.get_data_size()

    def get_lyrics(self, lyrics_id: int) -> str:
        return self.midnight.get_lyrics(lyrics_id)

    def get_genre(self, lyrics_id: int) -> str:
        return self.midnight.get_genre(lyrics_id)

    def get_pretrained_embedder(self) -> torch.nn.Embedding:
        return self.embedder.embeddings

    def get_posttrained_embedder(self) -> torch.nn.Embedding:
        return self.language_model.embedding

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def clean_text(self, text: str) -> str:
        return self.midnight.clean_text(text)

    def pollute_text(self, text: str) -> str:
        return self.midnight.pollute_text(text)

    def get_context_words(self, text, k=5):
        return self.midnight.get_context_words(text, k)

    def annotate_text(self, id: int, k=5) -> Annotation:
        return self.midnight.annotate_text(id, k)

    def tokenize_text(self, data: str) -> list[int]:
        return self.midnight.tokenize_text(data)

    def tokenize_genre(self, genre: str | list[str]) -> int | list[int]:
        return self.midnight.tokenize_text(genre)

    def detokenize_ids(self, data: list[int]) -> list[str]:
        return self.midnight.detokenize_ids(data)

    def embed_tokens(self, data):
        return self.midnight.embed_tokens(data)

    @torch.no_grad()
    def get_logits(self,
                   data: 'list[Sample] | list[tuple[str, str, str]]',
                   ) -> torch.Tensor:
        if False and isinstance(data[0], 'Sample'):
            genres = [x.annotation.genre for x in data]
            context_words = [" ".join(x.annotation.keywords) for x in data]
            lyrics = [x.lyrics for x in data]
        else:
            genres, context_words, lyrics = map(list, zip(*data))

        language_model = self.get_language_model()
        language_model.eval()
        device = next(language_model.parameters()).device

        annotations = self.tokenize_text([g + " " + c for g, c in zip(genres, context_words)])
        lyrics = self.tokenize_text(lyrics)
        annotations, lyrics = pad_lists(annotations), pad_lists(lyrics)

        annotations = torch.tensor(annotations, dtype=torch.long, device=device)
        lyrics = torch.tensor(lyrics, dtype=torch.long, device=device)

        return self.language_model(annotations, lyrics, True)

    @torch.no_grad()
    def bulk_inference(self,
                       genres: str | list[str],
                       context_words: str | list[str],
                       starting_words: str | list[str] = "",
                       starting_token="<SONG_START>", end_token="<SONG_END>",
                       max_len=200, n_songs: int = None,
                       temperature=1.0, top_k=50,
                       _temperature_epsilon=1e-4,
                       ) -> list[str]:
        # Data Sanity Checks

        if n_songs is None:
            if isinstance(genres, list): n_songs = len(genres)
            elif isinstance(context_words, list): n_songs = len(context_words)
            elif isinstance(starting_words, list): n_songs = len(starting_words)
            else: raise ValueError("Please provide either a list of genres, a list of context words, a list of starting words, or n_songs.")

        if isinstance(genres, str): genres = [genres] * n_songs
        if isinstance(context_words, str): context_words = [context_words] * n_songs
        if isinstance(starting_words, str): starting_words = [starting_words] * n_songs

        assert len(genres) == len(context_words) == len(starting_words)

        # Retrieving the language model

        language_model = self.get_language_model()
        device = next(language_model.parameters()).device
        language_model.eval()

        # Some helper methods

        def sample_batch(logits: torch.Tensor) -> torch.Tensor:
            """
            Sample one next token per row.  Returns shape (B, 1).

            Temperature fix: below TEMP_EPS we fall back to argmax (greedy),
            which avoids the softmax overflow that crashes the GPU at tiny temps.
            The top-k mask uses scatter so it never materializes a huge intermediate.
            """
            if temperature < _temperature_epsilon:
                return logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            scaled = logits / temperature

            # Top-k: zero out everything outside the k best positions
            k = min(top_k, scaled.size(-1))
            top_k_vals, top_k_idx = torch.topk(scaled, k, dim=-1)  # (B, k)
            # Build an -inf mask and scatter the top-k values back in
            filtered = torch.full_like(scaled, float('-inf'))
            filtered.scatter_(1, top_k_idx, top_k_vals)  # (B, V)

            probs = F.softmax(filtered, dim=-1)
            return torch.multinomial(probs, num_samples=1)  # (B, 1)

        # A new title

        starting_words = [(starting_token if s == '' else starting_token + " " + s) for s in starting_words]

        starting_ids = self.tokenize_text(starting_words)
        annotation_ids = self.tokenize_text([g + " " + c for g, c in zip(genres, context_words)])
        starting_ids, annotation_ids = pad_lists(starting_ids), pad_lists(annotation_ids)

        input_ids = torch.tensor(starting_ids, device=device)
        annotation_ids = torch.tensor(annotation_ids, device=device)

        for _ in range(max_len):
            preds = self.language_model(annotation_ids, input_ids)
            next_token = sample_batch(preds)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        input_ids = input_ids.cpu().tolist()
        generated_songs = self.detokenize_ids(input_ids)
        # generated_songs = [(song if end_token not in song else song.split(end_token, 1)[0]) for song in generated_songs]
        return list(map(self.pollute_text, generated_songs))
