import sentencepiece as spm
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from generator_core import *
from .conditional_lstm_lm import ConditionalLSTMLM, ConditionalDataset
from .word2vec import Word2Vec_SkipGram, ArrayToDatasetForW2V

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


class Midnight(Solution):

    def __init__(self, ds_data: pd.DataFrame, skip_model_loading=False):
        """
        Initialize all mid-state helpers
        """

        self.custom_tokens = set()
        self.genre_to_id = dict()

        self.ds_data = self._prepare_ds_data(ds_data)
        self.custom_tokens = self._get_custom_tokens()
        self.genre_to_id = self._get_genre_dict()
        self.id_to_genre = {v: k for k, v in self.genre_to_id.items()}
        self.tfidf = self._prepare_tfidf()
        self.feature_names = self.tfidf.get_feature_names_out()
        self.vocabulary = self._prepare_vocabulary()
        self.embedder = self._prepare_embedder()

        if not skip_model_loading:
            self.language_model = self._prepare_language_model()

    @cached()
    def _prepare_ds_data(self, ds_data):
        ds_data = ds_data.copy()
        ds_data['lyrics'] = ds_data['lyrics'].apply(self.clean_text)
        return ds_data

    @cached()
    def _get_custom_tokens(self):
        return self.custom_tokens

    @cached()
    def _get_genre_dict(self):
        return {g: i for i, g in enumerate(self.ds_data['tag'].unique().tolist())}

    @staticmethod
    def tokenize_for_tfidf(text):
        # VERIFY: Specify self.custom_tokens
        # VERIFY: Prevent keywords=['cheek', '?', 'let', ']', '[']
        return [token for token in text.split(' ') if re.match(r"[\w']+", token)]

    @cached()
    def _prepare_tfidf(self):
        tfidf = TfidfVectorizer(
            max_df=0.5,  # ignore very common words
            min_df=5,  # ignore rare words
            stop_words='english',
            max_features=50000,  # limit vocab size TODO: Optimize for this value
            token_pattern=None,
            tokenizer=Midnight.tokenize_for_tfidf,
        )
        tfidf.fit(self.ds_data['lyrics'])
        return tfidf

    def _prepare_vocabulary(self):
        # VERIFY: Specify custom_tokens
        spm_file = os.path.join('temp', 'lyrics_sp.model')
        if not os.path.exists(spm_file):
            temp_save = os.path.join('temp', 'lyrics_clean.csv')
            with open(os.path.join('temp', 'lyrics_clean.csv'), 'w', encoding='utf-8') as f:
                for lyric in self.ds_data['lyrics']:
                    f.write(lyric + '\n')
            spm.SentencePieceTrainer.Train(
                input=temp_save,
                model_prefix=spm_file.replace('.model', ''),
                vocab_size=16000,
                model_type='unigram',
                character_coverage=0.999,
                user_defined_symbols=list(self.custom_tokens)
            )
            os.remove(temp_save)
        else:
            print("Loaded Cache for Midnight._prepare_vocabulary", spm_file)
        sp = spm.SentencePieceProcessor()
        sp.Load(spm_file)
        return sp

    @cached()
    def _prepare_embedder(self):
        word2vec = Word2Vec_SkipGram(
            text_to_ids=self.tokenize_text,
            vocab_size=self.vocabulary.vocab_size(),
            d_embeds=512,
            max_norm=None,
        )
        word2vec.prepare_train(ArrayToDatasetForW2V(self.ds_data['lyrics']))
        return word2vec

    @cached()
    def _prepare_language_model(self):
        lstm = ConditionalLSTMLM(
            vocab_size=self.vocabulary.vocab_size(),
            embedding_dim=512,
            hidden_size=384,
            num_layers=2,
            num_genres=len(self.genre_to_id),
            genre_emb_dim=64,
            word2vec_weights=self.embedder.embeddings.weight,
            word2vec_frozen=False,
        )
        lstm.prepare_train(ConditionalDataset(self))
        return lstm

    def get_data_size(self) -> int:
        return self.ds_data.shape[0]

    def get_lyrics(self, lyrics_id: int) -> str:
        return self.ds_data.iloc[lyrics_id]['lyrics']

    def get_genre(self, lyrics_id: int) -> str:
        return self.ds_data.iloc[lyrics_id]['tag']

    def get_pretrained_embedder(self) -> torch.nn.Embedding:
        return self.embedder.embeddings

    def get_posttrained_embedder(self) -> torch.nn.Embedding:
        return self.language_model.embedding

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def clean_text(self, text: str) -> str:
        text = text.strip().lower()  # To lower case
        text = text.replace('\r\n', '\n')  # CRLF To LF
        text = re.sub(r"[\"“”‘’]", ' " ', text)  # Remove quotes
        text = re.sub(r"([\n.,!?;:\-])", r" \1 ", text)  # Add spaces around punctuations
        text = re.sub(r" +", " ", text).strip()  # Remove extra spaces

        text_items = []
        for line in text.split('\n'):
            line = line.strip()
            if len(line) > 40:
                text_items.append(line.strip())
                continue
            if len(line) == 0: continue

            consume = False
            for word in sure_fire_keywords:
                if word in line and re.search(rf'\b{word}\b', line):
                    token = f"<{word.upper()}>"
                    text_items.append(token)
                    if re.match(rf'^{word}\s*:.+', line):
                        text_items.append(line.split(":", 2)[1].strip())
                    self.custom_tokens.add(token)
                    consume = True
                    break

            if not consume:
                for word in two_fire_keywords:
                    if word in line and re.search(rf'\b{word}\b', line):
                        if any([token in line for token in structure_tokens]):
                            token = f"<{word.upper()}>"
                            text_items.append(token)
                            self.custom_tokens.add(token)
                            consume = True
                            break

            if not consume:
                text_items.append(line.strip())

        text = " <NEW_LINE> ".join(text_items)
        text = re.sub(r"([^\w<>'])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text)
        text = '<SONG_START> ' + text + ' <SONG_END>'
        self.custom_tokens.update({'<SONG_START>', '<SONG_END>', '<NEW_LINE>'})
        return text

    def pollute_text(self, text: str) -> str:
        text = text.replace(' <NEW_LINE> ', '\n')
        text = re.sub(r"<[\w ']+> ?", "", text)
        return text

    def _get_top_k_words(self, row, k=5):
        row_data = row.toarray().flatten()
        top_indices = np.argsort(row_data)[-k:]
        return [self.feature_names[i] for i in top_indices if row_data[i] > 0]

    def get_context_words(self, data: Unknown, k=5):
        lyrics = self._get_lyrics(data)
        return self._get_top_k_words(self.tfidf.transform([lyrics]), k=k)

    def annotate_text(self, id: int, k=5) -> Annotation:
        text = self.ds_data.iloc[id]['lyrics']
        genre = self.ds_data.iloc[id]['tag']
        context_words = self.get_context_words(text, k=k)
        return Annotation(id, genre, context_words)

    def tokenize_text(self, data: str) -> list[int]:
        if isinstance(data, int): data = self.ds_data.iloc[data]['lyrics']
        return self.vocabulary.encode_as_ids(data)

    def tokenize_genre(self, genre: str | list[str]) -> int | list[int]:
        return list(map(self.genre_to_id.__getitem__, genre)) if isinstance(genre, list) else self.genre_to_id[genre]

    def detokenize_ids(self, data: list[int]) -> list[str]:
        return self.vocabulary.decode_ids(data)

    @torch.no_grad()
    def embed_tokens(self, data):
        if isinstance(data, int): data = self.ds_data.iloc[data]['lyrics']
        if isinstance(data, str): data = self.tokenize_text(data)
        tokens = torch.tensor(data, dtype=torch.long)
        self.embedder = self.embedder.trainer.to('cpu')
        embeds = self.embedder.embeddings(tokens).numpy()
        return embeds

    @torch.no_grad()
    def get_logits(self,
                   data: list[Sample] | list[tuple[str, str, str]],
                   ) -> torch.Tensor:
        if isinstance(data[0], Sample):
            genres = self.tokenize_genre([x.annotation.genre for x in data])
            context_words = self.tokenize_text([" ".join(x.annotation.keywords) for x in data])
            lyrics = self.tokenize_text([x.lyrics for x in data])
        else:
            genres, context_words, lyrics = map(list, zip(*data))
            genres = self.tokenize_genre(genres)
            context_words = self.tokenize_text(context_words)
            lyrics = self.tokenize_text(lyrics)

        lengths = list(map(len, lyrics))
        context_words = pad_lists(context_words, fill_value=0)
        lyrics = pad_lists(lyrics, fill_value=0)

        language_model = self.get_language_model()
        language_model.eval()
        device = next(language_model.parameters()).device

        genres = torch.tensor(genres, dtype=torch.long, device=device)
        context_words = torch.tensor(context_words, dtype=torch.long, device=device)
        lyrics = torch.tensor(lyrics, dtype=torch.long, device=device)
        lengths = torch.tensor(lengths, dtype=torch.long, device=device)

        # print(lyrics.shape, context_words.shape, lengths.shape, genres.shape)

        return self.language_model(lyrics, context_words, lengths, genres)

    @torch.no_grad()
    def inference(self,
                  genre: str, context_words: list[str],
                  starting_words="",
                  starting_token="<SONG_START>", end_token="<SONG_END>",
                  max_len=200, temperature=1.0, top_k=50,
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
        genre_id = self.genre_to_id[genre]
        context_ids = self.tokenize_text(" ".join(context_words))

        device = next(self.language_model.parameters()).device
        input_ids = torch.tensor(starting_ids, device=device).unsqueeze(0)
        ending_ids = torch.tensor(ending_ids, device=device)
        context_ids = torch.tensor(context_ids, device=device).unsqueeze(0)
        genre_id = torch.tensor([genre_id], device=device)

        self.language_model.eval()
        for _ in range(max_len):
            lengths = torch.tensor([input_ids.size(1)], device=device)

            preds = self.language_model(input_ids, context_ids, lengths, genre_id)
            preds = preds[:, -1, :] / temperature
            next_token = sample_top_k(preds.squeeze(0), k=top_k)

            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
            if ends_with(input_ids, ending_ids): break

        input_ids = input_ids.squeeze(0).tolist()
        return self.detokenize_ids(input_ids)

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

        def pad_ragged(token_lists: list[list[int]], pad_val: int = 0):
            """Right-pad a list of token lists into a (B, L) tensor."""
            max_token_length = max(map(len, token_lists))
            padded = torch.full((len(token_lists), max_token_length), pad_val, dtype=torch.long, device=device)
            lengths = torch.zeros(len(token_lists), dtype=torch.long)
            for i, t in enumerate(token_lists):
                padded[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
                lengths[i] = len(t)
            return padded, lengths

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

        # Tokenizing the inputs

        genre_ids = torch.tensor([self.genre_to_id[g] for g in genres],
                                 dtype=torch.long, device=device)  # (B,)
        ctx_token_lists = [self.tokenize_text(c) for c in context_words]
        ctx_padded, _ = pad_ragged(ctx_token_lists)  # (B, C)

        prefix_token_lists = [self.tokenize_text(starting_token + sw) for sw in starting_words]
        prefix_padded, pfx_lengths = pad_ragged(prefix_token_lists)  # (B, P)

        end_ids = torch.tensor(self.tokenize_text(end_token), dtype=torch.long, device=device)
        end_len = end_ids.size(0)

        # Initial steps

        ctx_mask = (ctx_padded != 0).unsqueeze(-1).float()  # (B, C, 1)
        ctx_emb = language_model.embedding(ctx_padded)  # (B, C, E)
        ctx_vec = (ctx_emb * ctx_mask).sum(1) / ctx_mask.sum(1).clamp(min=1.0)  # (B, E)

        genre_vec = language_model.genre_embedding(genre_ids)  # (B, G)
        cond = torch.cat([ctx_vec, genre_vec], dim=-1)  # (B, E+G)

        h0 = torch.tanh(language_model.condition_proj_h(cond))  # (B, H)
        c0 = torch.tanh(language_model.condition_proj_c(cond))  # (B, H)

        L = language_model.lstm.num_layers
        hidden = (
            h0.unsqueeze(0).expand(L, n_songs, -1).contiguous(),
            c0.unsqueeze(0).expand(L, n_songs, -1).contiguous(),
        )

        # Hidden state update

        prefix_emb = language_model.embedding(prefix_padded)  # (B, P, E)
        packed_pfx = pack_padded_sequence(
            prefix_emb, pfx_lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        pfx_out_packed, hidden = language_model.lstm(packed_pfx, hidden)
        pfx_out, _ = pad_packed_sequence(pfx_out_packed, batch_first=True)  # (B, P, H)

        # Logits at the *last real* prefix token — this seeds the first generation step.
        last_pfx_pos = (pfx_lengths - 1).clamp(min=0).to(device)  # (B,)
        logits = language_model.output_layer(pfx_out[torch.arange(n_songs, device=device), last_pfx_pos])  # (B, V)

        # Decoder

        generated = prefix_padded.tolist()  # list[list[int]]
        finished = torch.zeros(n_songs, dtype=torch.bool, device=device)

        for _ in range(max_len):
            if finished.all():
                break

            next_tokens = sample_batch(logits)  # (B, 1)
            # Don't append anything meaningful for sequences that are already done.
            next_tokens = next_tokens.masked_fill(finished.unsqueeze(1), 0)

            for i in range(n_songs):
                if not finished[i]:
                    tok = next_tokens[i, 0].item()
                    generated[i].append(tok)
                    if (len(generated[i]) >= end_len and
                            torch.equal(
                                torch.tensor(generated[i][-end_len:], device=device),
                                end_ids,
                            )):
                        finished[i] = True

            # Single LSTM step with the carried hidden state
            emb = language_model.embedding(next_tokens)  # (B, 1, E)
            out, hidden = language_model.lstm(emb, hidden)  # (B, 1, H)
            logits = language_model.output_layer(out[:, 0, :])  # (B, V)

        return [self.pollute_text(self.detokenize_ids(seq)) for seq in generated]
