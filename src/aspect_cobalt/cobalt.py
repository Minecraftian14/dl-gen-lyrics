import torch.nn.functional as F
from torch import optim
from torch.utils import data

from aspect_midnight import Midnight
from aspect_tetra import BiGRULyricsModel, tetra_collate_fn, LyricsDataset
from dl_trainer import Trainer
from generator_core import *

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
        self.training_data = self._prepare_training_data()
        self.language_model = self._prepare_language_model()

    @cached()
    def _prepare_training_data(self, ds_data=None):
        if ds_data is None: ds_data = self.ds_data
        lyrics = ds_data['lyrics']
        genre = ds_data['tag'].map(lambda g: f"<genre_{g}>")
        context_words = ds_data['lyrics'].map(lambda l: " ".join([f"<theme_{t}>" for t in self.get_context_words(l)]))
        return genre.str.cat([context_words, lyrics], sep=" ").map(self.tokenize_text)

    def _optimizer(self, parameters):
        return optim.AdamW(parameters, lr=3e-4, weight_decay=1e-5)

    def _model_train_step(self, model, data): return model(data)

    def _model_criteria_step(self, criterion, preds, truth):
        preds = preds[0].permute(0, 2, 1)
        return criterion(preds, truth)

    def cache_checkpoint_callback(self, model_ref, epoch, iteration):
        import os
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        
        flight = 'Cobalt._prepare_language_model.cached'
        
        # Replicate your exact notebook logic to ensure 100% compatibility
        for file in ['bone', 'pkl']: # Added other possible extensions just in case
            file_path = os.path.join('temp', f'{flight}.{file}')
            if os.path.exists(file_path): os.remove(file_path)
            
        key_cached('cached', lambda: model_ref, group='Cobalt._prepare_language_model')
        print(f"Saved {flight} cache at epoch {epoch}, iteration {iteration}")
        
        # Evaluate BLEU on 1 batch so it's blazing fast
        model_ref.eval()
        with torch.no_grad():
            loader = model_ref.trainer.train_dataloader
            for input_ids, target_ids in loader:
                input_ids = input_ids.to(next(model_ref.parameters()).device)
                target_ids = target_ids.to(next(model_ref.parameters()).device)
                
                logits, _ = model_ref(input_ids)
                preds = logits.argmax(dim=-1)
                
                hypotheses_all = []
                references_all = []
                for b in range(preds.size(0)):
                    hypotheses_all.append([str(t.item()) for t in preds[b]])
                    references_all.append([[str(t.item()) for t in target_ids[b]]])
                    
                bleu = corpus_bleu(
                    references_all,
                    hypotheses_all,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=SmoothingFunction().method4,
                )
                print(f"Training BLEU Score (Epoch {epoch}, Iter {iteration}): {bleu:.4f}")
                break  # Stop after 1 batch
        model_ref.train()

    @cached()
    def _prepare_language_model(self):
        SAVE_CACHE_PERIODICALLY = True  # Toggle this True/False
        
        model = BiGRULyricsModel(
            vocab_size=self.vocabulary.vocab_size(),
            embed_dim=512,
            hidden_dim=512,
            num_layers=2,
            dropout=0.1,
            pad_id=0,
            word2vec_weights=self.embedder.embeddings.weight,
            word2vec_frozen=False,
        )
        dataloader = data.DataLoader(
            LyricsDataset(self.training_data),
            batch_size=20,
            shuffle=True,
            collate_fn=tetra_collate_fn,
        )

        model.trainer = Trainer(
            model=model,
            train_dataloader=dataloader,
            criterion=nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1),
            optimizer=self._optimizer,
            epochs=10,
            device='cpu',
            record_per_batch_training_loss=True,
            model_train_step=self._model_train_step,
            model_criteria_step=self._model_criteria_step,
            on_batch_callback=self.cache_checkpoint_callback if SAVE_CACHE_PERIODICALLY else None,
            on_batch_callback_frequency=50,
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
        return self.language_model.embedding

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def generate_evaluation_samples(self, output_file: str, n_songs: int = 5, max_len: int = 200, temperature: float = 0.8):
        """
        Generate multiple songs using the trained model and save them to a file.
        Each song is separated by 2 blank lines for LLM-as-a-judge evaluation.
        """
        import random
        
        # Ensure we have valid genres and sample context words from our dataset
        genres = list(self.ds_data['tag'].unique())
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(n_songs):
                # Pick a random genre and use random words as contextual themes
                genre = random.choice(genres)
                # Just get random keywords from a random song that matched this genre
                sample_text = self.ds_data[self.ds_data['tag'] == genre].sample(1).iloc[0]['lyrics']
                context_words = self.get_context_words(sample_text, k=3)
                
                print(f"Generating song {i+1}/{n_songs} [Genre: {genre}, Themes: {context_words}]...")
                
                tokens = self.inference(
                    genre=genre, 
                    context_words=context_words, 
                    max_len=max_len, 
                    temperature=temperature
                )
                
                # Format the generated string back to normal human text
                song_text = self.pollute_text(tokens)
                
                # Write to the evaluation file
                f.write(f"--- Song {i+1} ---\n")
                f.write(f"Genre: {genre}\n")
                f.write(f"Themes: {', '.join(context_words)}\n")
                f.write(f"Lyrics:\n{song_text}\n")
                f.write("\n\n\n")
                
        print(f"Successfully saved {n_songs} songs to {output_file}")

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
        genre_id = self.tokenize_text(f"<genre_{genre}>")
        context_ids = self.tokenize_text(" ".join([f"<theme_{t}>" for t in context_words]))
        annotation_ids = genre_id + context_ids

        starting_ids = annotation_ids + starting_ids

        device = next(self.language_model.parameters()).device
        input_ids = torch.tensor(starting_ids, device=device).unsqueeze(0)
        ending_ids = torch.tensor(ending_ids, device=device)

        self.language_model.eval()
        for _ in range(max_len):
            preds, _ = self.language_model(input_ids)
            preds = preds[0, -1]
            preds = preds / temperature
            next_token = sample_top_k(preds, k=top_k)

            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
            if ends_with(input_ids, ending_ids): break

        input_ids = input_ids.squeeze(0).tolist()
        return self.detokenize_ids(input_ids)

    @torch.no_grad()
    def get_logits(self,
                   data: list,  # list[Sample] | list[tuple[str, list[str], str]]
                   ) -> torch.Tensor:
        """
        Compute logits for a batch of (genre, context_words, lyrics) triples.

        Each item can be either a Sample namedtuple (with .annotation.genre /
        .annotation.keywords / .lyrics) or a plain tuple
        (genre: str, context_words: list[str] | str, lyrics: str).

        Genre and context are prepended as special tokens — matching how
        _prepare_training_data and inference encode inputs — so the BiGRU
        sees a single unified sequence per sample.

        Returns
        -------
        torch.Tensor  shape (B, L, V)
            Raw logits for every position in the (padded) sequence.
        """
        sequences = []
        for item in data:
            if isinstance(item, Sample):
                genre     = item.annotation.genre
                ctx_words = item.annotation.keywords  # list[str]
                lyrics    = item.lyrics
            else:
                genre, ctx_words, lyrics = item
                # Support passing a space-separated string instead of a list
                if isinstance(ctx_words, str):
                    ctx_words = ctx_words.split()

            genre_ids   = self.tokenize_text(f"<genre_{genre}>")
            context_ids = self.tokenize_text(" ".join([f"<theme_{t}>" for t in ctx_words]))
            lyrics_ids  = self.tokenize_text(lyrics)
            sequences.append(genre_ids + context_ids + lyrics_ids)

        # Right-pad sequences to the same length
        max_len = max(len(s) for s in sequences)
        device  = next(self.language_model.parameters()).device
        padded  = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        self.language_model.eval()
        logits, _ = self.language_model(padded)  # (B, L, V)
        return logits

    @torch.no_grad()
    def bulk_inference(self,
                       genres: "str | list[str]",
                       context_words: "str | list[str]",
                       starting_words: "str | list[str]" = "",
                       starting_token="<SONG_START>", end_token="<SONG_END>",
                       max_len=200, n_songs: int = None,
                       temperature=1.0, top_k=50,
                       ) -> list[str]:
        """
        Generate lyrics for a batch of songs simultaneously.

        Parameters
        ----------
        genres        : single genre string (broadcast) or one per song.
        context_words : space-separated theme words (broadcast) or one per song.
        starting_words: optional seed text appended after <SONG_START>.
        n_songs       : inferred from whichever argument is a list; required
                        only when all three are scalars.
        temperature   : softmax temperature (1.0 = unscaled).
        top_k         : vocabulary candidates sampled from per step.

        Returns
        -------
        list[str]  Decoded, polluted (human-readable) lyric strings.
        """
        # ── Data sanity checks ────────────────────────────────────────────────
        if n_songs is None:
            if isinstance(genres, list):          n_songs = len(genres)
            elif isinstance(context_words, list): n_songs = len(context_words)
            elif isinstance(starting_words, list): n_songs = len(starting_words)
            else:
                raise ValueError(
                    "Provide a list for at least one of genres / context_words / "
                    "starting_words, or pass n_songs explicitly."
                )

        if isinstance(genres, str):         genres        = [genres]        * n_songs
        if isinstance(context_words, str):  context_words = [context_words] * n_songs
        if isinstance(starting_words, str): starting_words = [starting_words] * n_songs

        assert len(genres) == len(context_words) == len(starting_words) == n_songs

        # ── Helpers ───────────────────────────────────────────────────────────
        def sample_top_k_batch(logits: torch.Tensor) -> torch.Tensor:
            """Top-k sample one token per row.  Returns shape (B, 1)."""
            logits = logits / max(temperature, 1e-8)
            k = min(top_k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)  # (B, k)
            probs   = F.softmax(top_k_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)             # (B, 1)
            return top_k_indices.gather(1, sampled)                       # (B, 1)

        device = next(self.language_model.parameters()).device
        self.language_model.eval()

        # ── Build per-song prefix: [genre] [context] <SONG_START> [seed] ─────
        prefix_token_lists = []
        for genre, ctx, sw in zip(genres, context_words, starting_words):
            ctx_list    = ctx.split() if isinstance(ctx, str) else ctx
            genre_ids   = self.tokenize_text(f"<genre_{genre}>")
            context_ids = self.tokenize_text(" ".join([f"<theme_{t}>" for t in ctx_list]))
            start_ids   = self.tokenize_text(starting_token + sw)
            prefix_token_lists.append(genre_ids + context_ids + start_ids)

        end_ids = self.tokenize_text(end_token)
        end_len = len(end_ids)

        # Right-pad all prefixes into a single (B, P) tensor
        max_prefix_len = max(len(p) for p in prefix_token_lists)
        input_ids = torch.zeros(n_songs, max_prefix_len, dtype=torch.long, device=device)
        for i, p in enumerate(prefix_token_lists):
            input_ids[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)

        # Track generated tokens and which sequences have hit <SONG_END>
        generated = [list(p) for p in prefix_token_lists]
        finished  = [False] * n_songs

        # ── Autoregressive decoding (full-sequence re-feed, matches inference) ─
        for _ in range(max_len):
            if all(finished):
                break

            preds, _ = self.language_model(input_ids)  # (B, L, V)
            last_logits = preds[:, -1, :]              # (B, V) — last position
            next_tokens = sample_top_k_batch(last_logits)  # (B, 1)

            # Append new column; zero out rows that are already done
            next_col = torch.zeros(n_songs, 1, dtype=torch.long, device=device)
            for i in range(n_songs):
                if not finished[i]:
                    tok = next_tokens[i, 0].item()
                    next_col[i, 0] = tok
                    generated[i].append(tok)
                    # Stop as soon as the end token appears at the tail
                    if (len(generated[i]) >= end_len and
                            generated[i][-end_len:] == end_ids):
                        finished[i] = True

            input_ids = torch.cat([input_ids, next_col], dim=1)  # grow by one column

        return [self.pollute_text(self.detokenize_ids(seq)) for seq in generated]