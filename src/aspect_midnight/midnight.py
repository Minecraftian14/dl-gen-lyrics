import re
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sentencepiece as spm

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from generator_core import *
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


# self.vocabulary = SimpleVocabulary({
# "<SONG_START>",
# "<SONG_END>",
# "<NEW_LINE>",
# "<INTRO>",
# "<VERSE>",
# "<CHORUS>",
# "<HOOK>",
# "<BUILD>",  # "<Pre-Hook>",
# })

class Midnight(Solution):

    def __init__(self, ds_data: pd.DataFrame):
        """
        Initialize all mid-state helpers
        """

        self.custom_tokens = set()
        self.ds_data = self._prepare_ds_data(ds_data)
        self.custom_tokens = self._get_custom_tokens()
        self.tfidf = self._prepare_tfidf()
        self.feature_names = self.tfidf.get_feature_names_out()
        self.vocabulary = self._prepare_vocabulary()
        self.embedder = self._prepare_embedder()

    @cached()
    def _prepare_ds_data(self, ds_data):
        ds_data = ds_data.copy()
        ds_data['lyrics'] = ds_data['lyrics'].apply(self.clean_text)
        return ds_data

    @cached()
    def _get_custom_tokens(self):
        return self.custom_tokens

    @staticmethod
    def tokenize_for_tfidf(text):
        return text.split(' ')

    @cached()
    def _prepare_tfidf(self):
        # TODO: Specify self.custom_tokens and self.feature_names (force as tokens)
        # TODO: Prevent keywords=['cheek', '?', 'let', ']', '[']
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
        # TODO: Specify custom_tokens
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
            )
            os.remove(temp_save)
        else:
            print("MN Cache Loaded:", spm_file)
        sp = spm.SentencePieceProcessor()
        sp.load(spm_file)
        return sp

    @cached()
    def _prepare_embedder(self):
        word2vec = Word2Vec_SkipGram(
            text_to_ids=self.tokenize_text,
            vocab_size=self.vocabulary.vocab_size(),
            d_embeds=300,
            max_norm=1.0,
        )
        word2vec.prepare_train(ArrayToDatasetForW2V(self.ds_data['lyrics']))
        word2vec.trainer.dataset_fraction = 30
        word2vec.train_model()
        return word2vec

    def load_dataset(self) -> CSVDatasetStreamer | list[CSVDatasetStreamer]:
        return None

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
        # text = re.sub(r"(?=[^\w<>'])", " ", text)
        # text = re.sub(r"(?<=[^\w<>'])", " ", text)
        text = re.sub(r"([^\w<>'])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text)
        # TODO: text = '<SONG_START>' + text + '<SONG_END>'
        #  And also add to self.custom_tokens
        return text

    def get_top_k_words(self, row, k=5):
        row_data = row.toarray().flatten()
        top_indices = np.argsort(row_data)[-k:]
        return [self.feature_names[i] for i in top_indices if row_data[i] > 0]

    def annotate_text(self, id: int) -> Annotation:
        text = self.ds_data.iloc[id]['lyrics']
        genre = self.ds_data.iloc[id]['tag']
        context_words = self.get_top_k_words(self.tfidf.transform([text]))
        return Annotation(id, genre, context_words)

    def tokenize_text(self, text: str) -> list[str]:
        if isinstance(text, int): text = self.ds_data.iloc[text]['lyrics']
        return self.vocabulary.encode_as_ids(text)

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
        return M2OLSTM(len(self.vocabulary))

    def sample_to_training_data(self, sample: 'Sample') -> 'Generator[TrainingData]':
        sample = torch.asarray(sample, dtype=torch.long)
        for i in range(1, len(sample)):
            yield sample[:i], sample[i]

    def collate_batch(self, batch: list['TrainingData']) -> 'BatchedTrainingData':
        xs, ys = list(zip(*batch))
        lengths = torch.tensor([len(x) for x in xs])
        xs = pad_sequence(xs, batch_first=True, padding_value=0)
        ys = torch.stack(ys)
        return xs, lengths, ys

    def train(self, model: 'nn.Module', sample: 'Sample'):
        super().train(model, sample)

    def evaluate(self, model: 'nn.Module', sample: 'Sample'):
        super().generate(model, sample)
