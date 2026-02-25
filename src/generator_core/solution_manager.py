from generator_core import *
from collections.abc import Callable
import numpy as np





class SolutionDeprecated:
    def __init__(self):
        self.stream: DatasetStreamer

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
