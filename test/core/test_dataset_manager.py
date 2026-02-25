from generator_core import *


def test_dataset_streamer():
    print("hola")
    stream = DatasetStreamer("movies_meta.csv")
    header = stream.header()
    stream.filter = lambda entry: entry[7] == 'en'
    stream.mapper = create_mapper(header, ['title', 'genres', 'imdb_id'])

    count = 0
    for _ in stream.stream():
        print(_)
        break
        count += 1

    print(count)

    en_stream = stream.cache("temp/movies_meta_en.csv")
    print(en_stream.header())
