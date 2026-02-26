from generator_core import *
import re
import os
import torch
from torch import nn








class Turquoise(Solution):

    def __init__(self):
        self.vocabulary = SimpleVocabulary({"<SONG_START>", "<SONG_END>", "<LINE>", "<STANZA>"})

    def load_dataset(self) -> CSVDatasetStreamer | list[CSVDatasetStreamer]:
        descriptor = LocalDatasetDescriptor("moosehead_lyrics_beatles")
        self.streamer = CSVDatasetStreamer(descriptor)
        if not descriptor.exists():
            with descriptor.open(mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['title', 'artist', 'content', 'genre'])
                for line in moosehead_lyrics_streamer.stream():
                    writer.writerow(line)
        return self.streamer

    def clean_text(self, text: str) -> str:
        text = text.strip().lower()  # To lower case
        text = text.replace('\r\n', '\n')  # CRLF To LF
        text = re.sub(r"[\"“”‘’]", "", text)  # Remove quotes
        text = re.sub(r"([\n.,!?;:()\-])", r" \1 ", text)  # Add spaces around punctuations
        text = re.sub(r" +", " ", text).strip()  # Remove extra spaces
        return text

    def annotate_text(self, text: str) -> 'Annotation':
        return None

    def tokenize_text(self, text: str) -> list[str]:
        # Special tokens
        text = re.sub(r'\n{2,}', '<STANZA>', text)
        text = re.sub(r'\n', '<LINE>', text)
        tokens = ['<SONG_START>'] + text.split() + ['<SONG_END>']
        return tokens

    def build_vocabulary(self, token: str) -> 'Vocabulary':
        self.vocabulary.build_vocabulary(token)
        return self.vocabulary

    def prepare_embedder(self, tokens: np.ndarray) -> np.ndarray:
        # We want to train the embeddings along with the model,
        # therefore, we return None, to tell the model that
        # indices is the data to be fed to the model during training.
        return None

    def embed_tokens(self, tokens: np.ndarray) -> np.ndarray:
        # We do not have an embedder, therefore, we return the tokens (indices) as is.
        # We will train an embedder in the model itself.
        return tokens

    def inject_sample(self, embeds: np.ndarray, annotation: Annotation) -> 'Sample':
        # We are not using annotations, therefore, we return the embeds as is.
        return embeds

    def prepare_model(self) -> nn.Module:
        encoder = _Encoder(len(self.vocabulary))
        decoder = _Decoder(len(self.vocabulary))
        return SimpleEncoderDecoderModel(self.vocabulary.encode("<SONG_START>"), encoder, decoder, 'cpu')

    def train(self, model: 'nn.Module', sample: 'Sample'):
        super().train(model, sample)

    def evaluate(self, model: 'nn.Module', sample: 'Sample'):
        super().generate(model, sample)
