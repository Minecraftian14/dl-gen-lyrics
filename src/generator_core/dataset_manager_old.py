from importlib.resources import files
import csv

resources = files('generator_core').joinpath('resources')

catalogue = {
    'genius_lyrics.csv',
    'moosehead_lyrics.csv',
    'movies_meta.csv',
    'movies_subtitles.csv',
}


def create_mapper(header, columns):
    x = []
    for i, h in enumerate(header):
        if h in columns:
            x.append(i)

    def mapper_op(entry):
        new_entry = []
        for i in x:
            new_entry.append(entry[i])
        return new_entry

    return mapper_op


class DatasetStreamer:
    def __init__(self, source, filter_op=lambda entry: True, mapper_op=lambda entry: entry, external=False):
        self.source = source
        self.external = external
        self.filter = filter_op
        self.mapper = mapper_op

    def __open__(self):
        if self.external:
            return open(self.source, encoding='utf-8')
        return resources.joinpath(self.source).open(encoding='utf-8')

    def header(self):
        if isinstance(self.source, DatasetStreamer): return self.source.header()
        with self.__open__() as file:
            return next(csv.reader(file))

    def stream(self):
        if isinstance(self.source, DatasetStreamer): return self.source.stream()
        with self.__open__() as file:
            reader = csv.reader(file)
            next(reader)
            for line in reader:
                if self.filter(line):
                    yield self.mapper(line)

    def keys(self):
        header = self.header()
        return {k: i for i, k in enumerate(header)}

    def cache(self, csv_file):
        with open(csv_file, 'w', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(self.mapper(self.header()))
            for line in self.stream():
                writer.writerow(line)
        return DatasetStreamer(csv_file, external=True)

    def window(self, window_size=5):
        window = []
        stream = self.stream()
        for i in range(window_size):
            window.append(next(stream))
        for line in stream:
            yield window
            window.pop(0)
            window.append(line)

    def flatten(self):
        for line in self.stream():
            yield from line

    def flattened(self):
        return DatasetStreamer(self.flatten)


    def filtered(self, filter_op):
        return DatasetStreamer(self, filter_op)

    def mapped(self, mapper_op):
        return DatasetStreamer(self, mapper_op=mapper_op)
