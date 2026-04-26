import sentencepiece as spm
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.utils.rnn import pad_sequence

from generator_core import *
from aspect_midnight import Word2Vec_SkipGram, ArrayToDatasetForW2V, Midnight

from .transformer_lm import TransformerModel, TransformerDataset

from dl_trainer import Trainer

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


class Teal(Solution):

    def __init__(self, ds_data: pd.DataFrame):
        """
        Initialize all mid-state helpers
        """

        self.midnight = Midnight(ds_data, skip_model_loading=True)

        self.ds_data = self.midnight.ds_data
        self.custom_tokens = self._get_custom_tokens()
        self.custom_tokens.add('<NEW_LINE>')
        self.genre_to_id = self.midnight.genre_to_id
        self.id_to_genre = {v: k for k, v in self.genre_to_id.items()}
        self.tfidf = self._prepare_tfidf()
        self.feature_names = self.tfidf.get_feature_names_out()

        self.vocabulary = self._prepare_vocabulary()
        self.embedder = self._prepare_embedder()
        self.language_model = self._prepare_language_model()

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
            tokenizer=Teal.tokenize_for_tfidf,
        )
        tfidf.fit(self.ds_data['lyrics'])
        return tfidf

    def _prepare_vocabulary(self):
        # VERIFY: Specify custom_tokens
        spm_file = os.path.join('temp', 'Teal.lyrics_sp.model')
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


    def model_train_step(self, model, data):
        return model(data[0])

    def model_criteria_step(self, criterion, preds, truth):
        preds = preds.permute(0, 2, 1)  # (B, V, T)
        return criterion(preds, truth)

    @cached()
    def _prepare_language_model(self):
        config = {
            "d_model": 512,
            "n_heads": 4,
            "n_groups": 2,
            "n_layers": 4

        }

        model = TransformerModel(
            vocab_size=self.vocabulary.vocab_size(),
            config=config,
            embedding_weights=self.embedder.embeddings.weight
        )

        dataset = TransformerDataset(self)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            collate_fn=dataset.collate_fn
        )

        model.trainer = Trainer(
            model=model,
            train_dataloader=dataloader,
            criterion=torch.nn.CrossEntropyLoss(ignore_index=0),
            model_train_step=self.model_train_step,
            model_criteria_step=self.model_criteria_step,
            epochs=20,
            device='cuda',
            record_per_batch_training_loss=True,
            checkpoint_frequency_batch=10,
        )

        return model

    def get_data_size(self) -> int:
        return self.ds_data.shape[0]

    def get_lyrics(self, lyrics_id: int) -> str:
        return self.ds_data.iloc[lyrics_id]['lyrics']

    def get_genre(self, lyrics_id: int) -> str:
        return self.ds_data.iloc[lyrics_id]['tag']

    def get_pretrained_embedder(self) -> torch.nn.Embedding:
        return self.embedder.embeddings

    def get_posttrained_embedder(self) -> torch.nn.Embedding:
        return self.language_model.embed

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
        self.custom_tokens.update({'<SONG_START>', '<SONG_END>'})
        return text

    def pollute_text(self, text: str) -> str:
        text = text.replace(' <NEW_LINE> ', '\n')
        text = re.sub(r" <[\w ']+>", "", text)
        return text

    def _get_top_k_words(self, row, k=5):
        row_data = row.toarray().flatten()
        top_indices = np.argsort(row_data)[-k:]
        return [self.feature_names[i] for i in top_indices if row_data[i] > 0]

    def get_context_words(self, data, k=5):
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

    def tokenize_genre(self, data: str) -> list[int]:
        return self.vocabulary.encode_as_ids(data)

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
    def get_logits(self, data: list[str] | list[int]):
        if isinstance(data[0], Sample):
            genres = [x.annotation.genre for x in data]
            context_words = [" ".join(x.annotation.keywords) for x in data]
            lyrics = [x.lyrics for x in data]
        else:
            genres, context_words, lyrics = map(list, zip(*data))

        lyrics = [f"genre {g} {c} {l}" for g, c, l in zip(genres, context_words, lyrics)]
        lyrics = self.tokenize_text(lyrics)
        lyrics = pad_lists(lyrics, fill_value=0)

        language_model = self.get_language_model()
        language_model.eval()
        device = next(language_model.parameters()).device

        lyrics = torch.tensor(lyrics, dtype=torch.long, device=device)

        logits = language_model(lyrics)
        return logits

    @torch.no_grad()
    def inference(self,
                  genre: str, context_words: list[str],
                  starting_words="",
                  starting_token="<SONG_START>", end_token="<SONG_END>",
                  max_len=40, temperature=1, top_k=50, top_p=0.9, penalty=1.2,
                  ):
        def ends_with(sequence, pattern):
            seq_len, pat_len = sequence.size(1), pattern.size(0)
            if seq_len < pat_len: return False
            return torch.equal(sequence[0, seq_len - pat_len:], pattern)

        def sample_top_k_top_p(logits, prev_tokens, top_k=50, top_p=0.9, temperature=0.9, penalty=1.2):
            logits = logits / temperature

            for token in set(prev_tokens):
                logits[token] /= penalty

            probs = F.softmax(logits, dim=-1)

            # top-k
            if top_k is not None:
                top_k = min(top_k, probs.size(-1))
                probs_topk, indices_topk = torch.topk(probs, top_k)
            else:
                probs_topk, indices_topk = probs, torch.arange(len(probs), device=probs.device)

            # sort
            sorted_probs, sorted_indices = torch.sort(probs_topk, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # top-p cutoff
            cutoff = cumulative_probs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False

            sorted_probs[cutoff] = 0
            sorted_probs /= sorted_probs.sum()

            sampled_idx = torch.multinomial(sorted_probs, 1)
            return indices_topk[sorted_indices[sampled_idx]]

        genre_token = f"genre {genre}"
        cond_text = genre_token + " " + " ".join(context_words)

        starting_ids = self.tokenize_text(cond_text + " " + starting_token + " " + starting_words)
        # starting_ids = self.tokenize_text(starting_token + starting_words)
        ending_ids = self.tokenize_text(end_token)
        # genre_id = self.genre_to_id[genre]
        # context_ids = self.tokenize_text(" ".join(context_words))

        device = next(self.language_model.parameters()).device
        input_ids = torch.tensor(starting_ids, device=device).unsqueeze(0)
        ending_ids = torch.tensor(ending_ids, device=device)
        # context_ids = torch.tensor(context_ids, device=device).unsqueeze(0)
        # genre_id = torch.tensor([genre_id], device=device)

        self.language_model.eval()
        for _ in range(max_len):
            # lengths = torch.tensor([input_ids.size(1)], device=device)

            preds = self.language_model(input_ids)
            preds = preds[:, -1, :]
            logits = preds.squeeze(0)
            prev_tokens = input_ids.squeeze(0).tolist()

            next_token = sample_top_k_top_p(
                logits,
                prev_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                penalty=penalty
            )

            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
            if ends_with(input_ids, ending_ids): break

        input_ids = input_ids.squeeze(0).tolist()
        output_text = self.detokenize_ids(input_ids)

        # Find start token
        start_idx = output_text.find(starting_token)

        if start_idx != -1:
            output_text = output_text[start_idx:]

        return output_text
        # return self.detokenize_ids(input_ids)

    @torch.no_grad()
    def bulk_inference(self,
                       genres: str | list[str],
                       context_words: str | list[str],
                       starting_words: str | list[str] = "",
                       starting_token="<SONG_START>", end_token="<SONG_END>",
                       max_len=200, n_songs: int = None,
                       temperature=0.9, top_k=50, top_p=0.9, penalty=1.2,
                       solstice_cutoff=None,
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

        def sample_batch(logits: torch.Tensor, prev_tokens) -> torch.Tensor:
            """
            Sample one next token per row.  Returns shape (B, 1).

            Temperature fix: below TEMP_EPS we fall back to argmax (greedy),
            which avoids the softmax overflow that crashes the GPU at tiny temps.
            The top-k mask uses scatter so it never materializes a huge intermediate.
            """
            if temperature < _temperature_epsilon:
                return logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            logits = logits / temperature

            for token in prev_tokens:
                logits[:, token] /= penalty

            probs = F.softmax(logits, dim=-1)

            # top-k
            probs_topk, indices_topk = torch.topk(probs, min(top_k, probs.size(-1)))

            # sort
            sorted_probs, sorted_indices = torch.sort(probs_topk, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # top-p cutoff
            cutoff = cumulative_probs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False

            sorted_probs[cutoff] = 0
            sorted_probs /= sorted_probs.sum()

            sampled_idx = torch.multinomial(sorted_probs, 1)
            sampled_idx = [sorted_indices[i, x] for i, x in enumerate(sampled_idx)]
            sampled_idx = [indices_topk[i, x] for i, x in enumerate(sampled_idx)]
            return torch.tensor(sampled_idx, device=device, dtype=torch.long).unsqueeze(1)

        # A new title

        starting_words = [(starting_token if s == '' else starting_token + " " + s) for s in starting_words]

        starting_ids = self.tokenize_text(starting_words)
        annotation_ids = self.tokenize_text(["genre " + g + " " + c for g, c in zip(genres, context_words)])
        starting_ids = [a + s for s, a in zip(starting_ids, annotation_ids)]
        starting_ids = pad_lists(starting_ids)

        input_ids = torch.tensor(starting_ids, device=device)

        for _ in range(max_len):
            if solstice_cutoff is not None and input_ids.size(1) > solstice_cutoff:
                preds = self.language_model(input_ids[:, -solstice_cutoff:])
            else:
                preds = self.language_model(input_ids)
            next_token = sample_batch(preds[:, -1, :], set(input_ids.flatten().tolist()))

            input_ids = torch.cat([input_ids, next_token.view(-1, 1)], dim=1)

        input_ids = input_ids.cpu().tolist()
        generated_songs = self.detokenize_ids(input_ids)
        generated_songs = [(song if end_token not in song else song.split(end_token, 1)[0]) for song in generated_songs]
        return list(map(self.pollute_text, generated_songs))
