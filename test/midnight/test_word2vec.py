import random
import numpy as np
import matplotlib.pyplot as plt

import torch.utils.data as data

from aspect_midnight import Word2Vec_SkipGram, ArrayToDatasetForW2V


def tokenize(sentence: tuple[str]):
    return [float(ord(x) - ord(x)) for x in sentence.split(' ')]


def test_training_loop():
    words = "q w e r t y u i o p a s d f g h j k l z x c v b n m".split(' ')
    p = np.linspace(0, 1, len(words))
    p = p / np.sum(p)

    sentences = []
    for _ in range(100):
        sentences.append(" ".join(np.random.choice(words, random.randint(5, 10), p=p)))

    model = Word2Vec_SkipGram(vocab_size=len(words), text_to_ids=tokenize)
    model.prepare_train(ArrayToDatasetForW2V(sentences))
    model.train_model()
    plt.plot(model.trainer.loss['train.batch'])
