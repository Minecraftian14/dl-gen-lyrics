from dataclasses import dataclass

import numpy as np
import torch


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

    def get_data_size(self) -> int: ...

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

    def get_context_words(self, data:Unknown, k=5):
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

    def tokenize_genre(self, genre: str | list[str]) -> int | list[int]:
        """
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
    def get_logits(self,
                   data: 'list[Sample] | list[tuple[str, str, str]]',
                   ) -> torch.Tensor:
        """
        Generate logits for a given set of input samples or tuples, using a specified
        language model. This method tokenizes the inputs, pads sequences for batch
        processing, and computes the output logits.

        :param data: Input data to process. Can be a list of `Sample` objects or a
            list of tuples, where each tuple contains (genre, context_words, lyrics).
        :type data: list[Sample] | list[tuple[str, str, str]]

        :return: A numpy array of logits computed for the input samples.
        :rtype: np.ndarray
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
        Performs bulk inference to generate song lyrics based on given genres, context words, and starting
        words. This function supports batching and optional configuration to control the generation
        processes such as temperature and top-k sampling.

        :param genres: A string or list of strings, representing the genres of the songs to be generated. If a single string is provided, it will be broadcast across all songs.
        :param context_words: A string or list of strings as context inputs for generating lyrics. If a single string is provided, it will be broadcast across all songs.
        :param starting_words: A string or list of strings containing initial words to seed the generation process. Defaults to an empty string. A single string will be broadcast across all songs.
        :param starting_token: A special token marking the start of a song. Defaults to "<SONG_START>".
        :param end_token: A special token marking the end of a song. Defaults to "<SONG_END>".
        :param max_len: The maximum number of tokens each generated song can contain. Defaults to 200.
        :param n_songs: The total number of songs to generate. This parameter is automatically inferred based on the length of the provided lists for `genres`, `context_words`, or `starting_words`, if not explicitly set.
        :param temperature: A float parameter that is used to control the diversity of the generated text. Higher values lead to more random outputs, while lower values make the output more deterministic. Defaults to 1.0.
        :param top_k: Limits the sampling pool to the top-k tokens with the highest probabilities, for controlled and faster generation. Defaults to 50.
        :param _temperature_epsilon: A threshold that is used to avoid numerical instability when the temperature value is extremely low. Defaults to 1e-4. This parameter is internal and should not be modified.
        :return: A list of generated song lyrics as strings.
        :rtype: list[str]
        """
