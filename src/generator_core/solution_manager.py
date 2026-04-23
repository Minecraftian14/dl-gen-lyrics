from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


@dataclass
class Annotation:
    # Class used only for static typing purposes
    lyrics_id: int
    genre: str
    keywords: list[str]


@dataclass
class Sample:
    # Class used only for static typing purposes
    lyrics_id: int
    lyrics: str
    annotation: Annotation


Unknown = int | str | Annotation | Sample


class Solution:

    def get_lyrics(self, lyrics_id: int) -> str:
        """
        Fetches the lyrics for a given song identifier.

        It is expected that the clean text function was used to preprocess the dataset.
        Therefore, the returning lyrics being clean and custom tokens enriched.

        :param lyrics_id: The unique integer identifier of the song.
        :return: The lyrics of the song identified by the given ID.
        """

    def get_genre(self, lyrics_id: int) -> str:
        """
        Retrieve the genre of a song based on its lyrics ID.

        :param lyrics_id: The unique identifier for the song's lyrics.
        :return: The genre of the song associated with the given lyrics ID.
        """

    def get_pretrained_embedder(self) -> torch.nn.Embedding:
        """
        :return:
        """

    def get_posttrained_embedder(self) -> torch.nn.Embedding:
        """
        :return:
        """

    def get_language_model(self) -> torch.nn.Module:
        """
        :return:
        """

    def _get_id(self, data: Unknown):
        match data:
            case int(): return data
            case str(): return None
            case Annotation(): return data.lyrics_id
            case Sample(): return data.lyrics_id

    def _get_lyrics(self, data: Unknown):
        match data:
            case int(): return self.get_lyrics(data)
            case str(): return data
            case Annotation(): return self.get_lyrics(data.lyrics_id)
            case Sample(): return data.lyrics

    def _get_genre(self, data: Unknown):
        match data:
            case int(): return self.get_genre(data)
            case str(): return None
            case Annotation(): return self.get_genre(data.lyrics_id)
            case Sample(): return self.get_genre(data.lyrics_id)

    def clean_text(self, data: Unknown) -> str:
        """
        Given raw texts, implement this function to clean the text.
        This involves activities like:
        * Changing the text to lower case
        * Managing different OSs new line paradigm
        * Removing unnecessary special characters, especially the unicode stylistic ones
        * Removing extra spaces
        * This also involves enriching the text with custom tokens.
          That's... A tricky part, since you need to parse tokens like "Chorus:", "[CHORUS]", "[Chorus: Main Singer]" to "<CHORUS>"
        * Make sure to replace new lines with the special token "<NEW_LINE>"
        * Also, make sure to add the <SONG_START> and <SONG_END> tokens, if not already present.

        Another big responsibility of this function is to keep track of al the found custom tokens, in case recorded dynamically.
        Otherwise, not required.

        Ideally, it's better to have such things preprocessed, preventing unnecessary CPU cycles to clean text during training.
        Therefore, use this function during the dataset loading to replace the text with clean version.

        :param data: This can be anything... an int, or perhaps a string... Use self._get_lyrics(data) to get the lyrical text.
        :return: Cleaned text
        """

    def pollute_text(self, text: str) -> str:
        """
        Kind of like the opposite of clean_text.
        Basically, replace <NEW_LINE> with actual new lines and remove other structural tokens like song start and song end.

        :param text: Polluted text
        :return:
        """

    def get_context_words(self, text: Unknown, k=5) -> list[str]:
        """
        Retrieve context words around the provided sequence of words.
        A nice idea would be to train a TFIDF and use it to extract the most relevant words.

        :param text: A sequence of words from which the context is extracted.
        :param k: Number of words to extract before and after the central word. Defaults to 5.
        :return: A list of tuples words
        :rtype: list[str]
        """

    def annotate_text(self, data: Unknown, k=5) -> Annotation:
        """
        Annotates a text by extracting its genre and generating context words.

        This method takes the identifier of a text, extracts its corresponding
        lyrics and genre from the dataset, and computes a set of context words
        based on specified parameters. It then returns an Annotation object
        containing the extracted information.

        :param data: Data to get words from
        :param k: Number of context words to extract. Defaults to 5.
        :return: An Annotation object containing the text's ID, genre, and context words.
        """
        text = self._get_lyrics(data)
        genre = self._get_genre(data)
        context_words = self.get_context_words(text, k=k)
        return Annotation(self._get_id(data), genre, context_words)

    def tokenize_text(self, data: Unknown) -> list[int]:
        """
        Tokenizes a given piece of text into a list of integer IDs
        based on the predefined vocabulary.

        :param data: Input text data to be tokenized. Could be a single piece of text.
        :return: A list of integer IDs representing the tokenized text data.
        """

    def detokenize_ids(self, data: list[int]) -> str:
        """
        Detokenizes a list of token IDs into their corresponding string representations
        using the associated vocabulary. This function takes a list of integers representing
        token IDs and returns the decoded string.

        :param data: A list of integers which are token IDs to be converted into strings.
        :return: A string obtained by decoding the token IDs using the vocabulary.
        """

    def get_embedder_parameter_count(self):
        embedder = self.get_pretrained_embedder()
        return sum([parameter.numel() for parameter in embedder.parameters()])

    def get_language_model_parameter_count(self):
        language_model = self.get_language_model()
        embedder = self.get_posttrained_embedder()
        parameter_count = sum([parameter.numel() for parameter in language_model.parameters()])
        parameter_count -= sum([parameter.numel() for parameter in embedder.parameters()])
        return parameter_count

    @torch.no_grad()
    def embed_tokens(self, data: Unknown) -> np.ndarray:
        """
        Embeds input tokens into numerical vectors using the model's embedding layer.

        This method preprocesses the input data by tokenizing and converting
        it into tensors. The tokenized data is passed through the embedding layer
        of the model to produce corresponding embeddings in NumPy array format.

        :param data: Text input or lyrics data to be embedded.
        :return: A NumPy array of embeddings corresponding to the tokenized input.
        """

    @torch.no_grad()
    def inference(self,
                  genre: str, context_words: list[str],
                  starting_words="",
                  starting_token="<SONG_START>", end_token="<SONG_END>",
                  max_len=40, temperature=1.0, top_k=50,
                  ):
        """
        Generates a sequence of text based on the specified genre, context, and
        given starting words using a language model. The method uses sampling
        from the model with `top-k` sampling and temperature scaling.

        :param genre: The genre of the generated text. This should be a string corresponding to a valid genre recognized by the model.
        :param context_words: A list of additional context words or phrases to aid generation.
        :param starting_words: A string with words to initiate the generated text. Defaults to an empty string.
        :param starting_token: A special beginning token to mark the start of the text. Defaults to "<SONG_START>".
        :param end_token: A special ending token that signifies the end of the generated text. Defaults to "<SONG_END>".
        :param max_len: The maximum length for the generated sequence. Defaults to 40.
        :param temperature: The temperature scaling for sampling. Higher values generate more diverse text, while lower values result in more deterministic predictions. Defaults to 1.0.
        :param top_k: The number of top predictions to consider when sampling the next token. Defaults to 50.
        :return: A string containing the complete generated text sequence.
        """

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
        """
        """

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