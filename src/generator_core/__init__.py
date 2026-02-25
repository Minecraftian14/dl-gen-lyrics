from .dataset_manager_old import *
from .embed_manager import *
from .lyrics_manager import *
from .model_manager import *
from .trainer import *
from .word2vec import *

__init__ = [
    'DatasetStreamer',
    'catalogue',
    'create_mapper',


    'Solution',


    'Trainer',



    'SkipGramWord2Vec',



]

if __name__ == '__main__':
    print(globals())
    print(locals())
