import torch
import torch.nn.functional as F

from aspect_midnight import Midnight
from generator_core import *
from .encoder_decoder import EncoderDecoderLSTM, SlidingWindowDatasetTruncated

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
