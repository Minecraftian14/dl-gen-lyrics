from .impl.Timer import Timer, TypedTimer
from .impl.Trainer import Trainer
from .impl.Vocabulary import SimpleVocabulary

from .dataset_manager import *
from .embed_manager import *
from .lyrics_manager import *
from .model_manager import *
from .solution_manager import *
from .word2vec import *


__init__ = [
    'Timer',
    'TypedTimer',
    'Trainer',
    'SimpleVocabulary',


    'DatasetDescriptor'
    'LocalDatasetDescriptor'
    'CSVDatasetStreamer'

    'genius_lyrics',
    'moosehead_lyrics',

    'genius_lyrics_streamer',
    'moosehead_lyrics_streamer',
    'movies_subtitles_streamer',

    
    'Solution',


    'Trainer',


    'SkipGramWord2Vec',


]

if __name__ == '__main__':
    print(globals())
    print(locals())
