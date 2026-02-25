from generator_core import *
from generator_core.solution_manager import SolutionDeprecated
import re


class ABCDE(SolutionDeprecated):
    def step_1_tokenize(self, song: str) -> list[str]:

        # Simple sanitization
        song = song.lower().replace('\r\n', '\n')
        song = re.sub(r"[\"“”‘’]", "", song)
        song = re.sub(r"([.,!?;:()\-])", r" \1 ", song)
        song = re.sub(r"\s+", " ", song).strip()

        # Special tokens
        song = re.sub(r'\n{2,}', '<STANZA>', song)
        song = re.sub(r'\n', '<LINE>', song)
        song = ['<SONG_START>'] + song.split() + [['<SONG_END>']]

        return song

    def step_2_cache_dataset(self):
        stream = DatasetStreamer("moosehead_lyrics.csv")
        keys = stream.header()
        stream.filter = lambda entry: entry[keys['artist']] in ['The Beatles']
        stream.mapper = lambda entry: [" ".join(self.step_1_tokenize(entry[keys['text']]))]
        self.stream = stream.cache("temp/moosehead_lyrics_beatles.csv")

    def step_3_setup_embedder(self):
        super().step_3_setup_embedder()





