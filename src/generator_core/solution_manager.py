from .dataset_manager import CSVDatasetStreamer
from .impl.Trainer import Trainer

import numpy as np
from typing import Callable, Any, Generator
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader


class Annotation:
    # Class used only for static typing purposes
    text_id: str
    genre: list[str]
    keywords: list[str]


class Vocabulary:
    # Class used only for static typing purposes
    def encode(self, token: str) -> int: ...

    def decode(self, token: int) -> str: ...

    def __len__(self) -> int: ...


class Solution:

    def load_dataset(self) -> CSVDatasetStreamer | list[CSVDatasetStreamer]:
        """
        Override this method to load your dataset and store it in self.
        :return Either a singular streamer or multiple streamers
        """

    def clean_text(self, text: str) -> str:
        """
        :param text: The whole song or text
        :return: Clean/normalized text
        """

    def annotate_text(self, text: str) -> 'Annotation':
        """
        An annotation is represented as:
        ```
        {
            "text id": "Some kind of identifier or index",
            "genre": [ List of genres or theme of the text ],
            "keywords": [ List of keywords present in the text ]
        }
        ```
        :param text: The whole song or text (Usually cleaned)
        :return: A dictionary of annotations
        """

    def tokenize_text(self, text: str) -> list[str]:
        """
        :param text: The whole song or text (Usually cleaned)
        :return: Tokenized text
        """

    def build_vocabulary(self, token: str) -> 'Vocabulary':
        """
        This function will be called once for each token in the dataset.
        Use it to build up the vocabulary of your model.
        :return: Latest Vocabulary
        """

    def prepare_embedder(self, tokens: np.ndarray) -> np.ndarray:
        """
        A one-time preparation step for the embedder.
        Either load existing ones, extend existing ones, or train a completely new one.

        :param tokens: Tokens (Usually OHE-ed)
        :return: Embeds
        """

    def embed_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """
        :param tokens: Tokens (Usually OHE-ed)
        :return: Embeds
        """

    def inject_sample(self, embeds: np.ndarray, annotation: Annotation) -> 'Sample':
        """
        Additional note,
        Only the code written in your implementation will handle the Sample; therefore, no formal
        typing is required for this. If you are not working with annotations, return the embeds.
        If you are working with annotations, decide on a Conditioning Method; for example,
        - You can return tuples; like (embeds, annotations)
        - You can return combined embeds; like stacked (embeds, embedded-annotations)
        Since you'll be writing the code to handle it, it's up to you.

        :param embeds: Embeds for one text
        :param annotation: Annotations for that text
        :return: the sample according to your formulation of the Conditioning Method
        """

    def sample_to_training_data(self, sample: 'Sample') -> 'Generator[TrainingData]':
        """
        :param sample:
        :return:
        """

    def collate_batch(self, batch: list['TrainingData']) -> 'BatchedTrainingData':
        """
        :param batch:
        :return:
        """

    def prepare_model(self) -> nn.Module:
        """
        Load an existing model or create a new one.
        """

    def train(self, model: nn.Module, sample: 'Sample'):
        ...

    def evaluate(self, model: nn.Module, sample: 'Sample'):
        ...


class SolutionManager:
    def __init__(self,
                 solution_maker: Callable[[], Solution],
                 batch_size: int = 32,
                 ):
        self.solution_maker = solution_maker
        self.solution: Solution
        self.streamer: CSVDatasetStreamer
        self.vocabulary: Vocabulary
        self.model: nn.Module
        self.trainer: Trainer
        self.samples_cached = False
        self.batch_size: int
        self.dataset: IterableDataset = None
        self.loader: DataLoader = None
        self.set_batch_size(batch_size)

    def _stream_literal_tokens(self) -> 'Generator[str]':
        for title, artist, content, genre in self.streamer.stream():
            content = self.solution.clean_text(content)
            literal_tokens = self.solution.tokenize_text(content)
            for token in literal_tokens:
                yield token

    def _stream_samples(self) -> 'Generator[Sample]':
        for title, artist, content, genre in self.streamer.stream():
            content = self.solution.clean_text(content)
            annotation = self.solution.annotate_text(content)
            literal_tokens = self.solution.tokenize_text(content)
            indicial_tokens = np.zeros(len(literal_tokens))
            for i, token in enumerate(literal_tokens):
                indicial_tokens[i] = self.vocabulary.encode(token)
            embeds = self.solution.embed_tokens(indicial_tokens)
            sample = self.solution.inject_sample(embeds, annotation)
            yield sample

    def _stream_training_data(self) -> 'Generator[TrainingData]':
        for sample in self._stream_samples():
            training_data = self.solution.sample_to_training_data(sample)
            yield from training_data

    def _training_data_set(self):
        if self.dataset: return self.dataset

        class TrainingDataset(IterableDataset):
            def __init__(self, manager):
                self.manager: SolutionManager = manager

            def __iter__(self):
                for data in self.manager._stream_training_data():
                    yield data

        self.dataset = TrainingDataset(self)
        return self.dataset

    def _training_data_loader(self):
        if self.loader is not None: return self.loader

        self.loader = DataLoader(
            self._training_data_set(),
            batch_size=self.batch_size,
            collate_fn=self.solution.collate_batch,
        )
        return self.loader

    def prepare_solution(self):

        self.solution = self.solution_maker()
        self.streamer = self.solution.load_dataset()

        for literal_token in self._stream_literal_tokens():
            self.vocabulary = self.solution.build_vocabulary(literal_token)

        self.solution.prepare_embedder(None)
        self.model = self.solution.prepare_model()
        self.trainer = Trainer(
            model=self.model,
            train_dataloader=self._training_data_loader(),
            epochs=1,
            optimizer=lambda params: torch.optim.Adam(params, lr=0.003),
            model_train_step=lambda model, data: model(data[0], data[1]),
            device="cuda",
        )

    def cache_samples(self):
        ...

    def phase_2(self):
        ...

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.loader = None


class SolutionDeprecated:
    def __init__(self):
        self.stream: None

    def step_1_tokenize(self, sentence: str) -> list[str]:
        """
        Convert the thing to a list of words, tokens and whatever magic you are cooking
        """
        ...

    def step_2_cache_dataset(self):
        """
        Reads and caches a dataset and saves its stream locally
        """
        ...

    def step_3_setup_embedder(self):
        """
        Create or Load an existing embedder
        """
        ...

    def step_4_embed(self, tokens: list[str]) -> np.ndarray:
        """
        Embed the given stuff
        """
        ...

    def step_5_setup_model(self):
        """
        Create or Load an existing model
        """
        ...
