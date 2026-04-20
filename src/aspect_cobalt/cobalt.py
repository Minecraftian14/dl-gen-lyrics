import torch.nn.functional as F
from torch.utils import data

from aspect_midnight import Midnight
from generator_core import *
from .model.bigru import BiGRULyricsModel

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


class CobaltDataset(data.Dataset):
    def __init__(self, cobalt: 'Cobalt'):
        self.cobalt = cobalt

    def __len__(self):
        return len(self.cobalt.ds_data)

    def __getitem__(self, index):
        cobalt = self.cobalt
        sample = cobalt.ds_data.iloc[index]
        genre = sample['genre']
        context_words = sample['context_words']
        starting_words = sample['starting_words']
        ending_words = sample['ending_words']
        return genre, context_words, starting_words, ending_words


class Cobalt(Solution):

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

    def _create_sequence(self, sample: Sample):
        sequence = f"{sample.annotation.genre} {"".join(sample.annotation.keywords)} {sample.lyrics}"
        return sequence

    @cached()
    def _prepare_training_data(self, ds_data=None):
        if ds_data is None: ds_data = self.ds_data
        lyrics = ds_data['lyrics']
        genre = ds_data['tag'].map(lambda g: f"<genre_{g}>")
        context_words = ds_data['lyrics'].map(lambda l: " ".join([f"<theme_{t}>" for t in self.get_context_words(l)]))
        return genre.str.cat([context_words, lyrics], sep=" ")

    @cached()
    def _prepare_language_model(self):
        return BiGRULyricsModel(
            vocab_size=self.vocabulary.vocab_size(),
            embed_dim=512,
            hidden_dim=512,
            num_layers=2,
            dropout=0.3,
            pad_id=0,
            word2vec_weights=self.embedder.embeddings.weight,
            word2vec_frozen=True,
        )

    def clean_text(self, text: str) -> str:
        return self.midnight.clean_text(text)

    def pollute_text(self, text: str) -> str:
        return self.midnight.pollute_text(text)

    def get_context_words(self, text, k=5):
        return self.midnight.get_context_words(text, k=k)

    def annotate_text(self, id: int, k=5) -> Annotation:
        return self.midnight.annotate_text(id, k=k)

    def tokenize_text(self, data: str) -> list[int]:
        return self.midnight.tokenize_text(data)

    def detokenize_ids(self, data: list[int]) -> list[str]:
        return self.midnight.detokenize_ids(data)

    @torch.no_grad()
    def embed_tokens(self, data):
        return self.midnight.embed_tokens(data)

    @torch.no_grad()
    def inference(self,
                  genre: str, context_words: list[str],
                  starting_words="",
                  starting_token="<SONG_START>", end_token="<SONG_END>",
                  max_len=40, temperature=1.0, top_k=50,
                  ):
        def ends_with(sequence, pattern):
            seq_len, pat_len = sequence.size(1), pattern.size(0)
            if seq_len < pat_len: return False
            return torch.equal(sequence[0, seq_len - pat_len:], pattern)

        def sample_top_k(logits, k=50):
            k = min(k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, k)
            probs = F.softmax(top_k_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            return top_k_indices[sampled_idx]

        starting_ids = self.tokenize_text(starting_token + starting_words)
        ending_ids = self.tokenize_text(end_token)
        genre_id = self.tokenize_text(genre)
        context_ids = self.tokenize_text(" ".join(context_words))
        annotation_ids = genre_id + context_ids

        device = next(self.language_model.parameters()).device
        input_ids = torch.tensor(starting_ids, device=device).unsqueeze(0)
        ending_ids = torch.tensor(ending_ids, device=device)
        annotation_ids = torch.tensor([annotation_ids], device=device)

        self.language_model.eval()
        for _ in range(max_len):
            preds = self.language_model(annotation_ids, input_ids)
            preds = preds / temperature
            next_token = sample_top_k(preds.squeeze(0), k=top_k)

            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
            if ends_with(input_ids, ending_ids): break

        input_ids = input_ids.squeeze(0).tolist()
        return self.detokenize_ids(input_ids)
