from importlib.resources import files
from collections.abc import Iterator
import csv
import json5
import os
import sqlite3

resources = files('generator_core').joinpath('resources')


class DatasetDescriptor:
    def __init__(self,
                 file_name: str,
                 csv_header: list[str],
                 title: str,
                 artist: str,
                 content: str,
                 genre: str, ):
        self.file_name = f'{file_name}.csv'
        self.csv_header = csv_header
        self.title = title
        self.artist = artist
        self.content = content
        self.genre = genre

    def get_indices(self, header):
        return [header.index(self.title) if self.title in header else -1,
                header.index(self.artist) if self.artist in header else -1,
                header.index(self.content) if self.content in header else -1,
                header.index(self.genre) if self.genre in header else -1, ]

    def open(self):
        return resources.joinpath(self.file_name).open(encoding='utf-8')


class LocalDatasetDescriptor(DatasetDescriptor):
    def __init__(self, file_name: str):
        super().__init__(file_name, ['title', 'artist', 'content', 'genre'], 'title', 'artist', 'content', 'genre')
        self.file_path = os.path.join('temp', f'{file_name}.csv')

    def get_indices(self, header):
        return [0, 1, 2, 3]

    def open(self, **kwargs):
        return open(self.file_path, **{'encoding': 'utf-8', **kwargs})

    def exists(self):
        return os.path.exists(self.file_path)


class CSVDatasetStreamer:
    def __init__(self, descriptor: DatasetDescriptor):
        self.descriptor = descriptor

    def stream(self) -> Iterator[list[str]]:
        """
        :return: Iterable of lists of strings in [title, artist, content, genre] order
        """
        with self.descriptor.open() as file:
            reader = csv.reader(file)
            indices = self.descriptor.get_indices(next(reader))
            for line in reader:
                yield [line[i] if i > -1 else 'null' for i in indices]


genius_lyrics = DatasetDescriptor(
    "genius_lyrics",
    ["title", "tag", "artist", "year", "views", "features", "lyrics", "id", "language_cld3", "language_ft", "language"],
    "title",
    "artist",
    "lyrics",
    "tag", )
genius_lyrics_streamer = CSVDatasetStreamer(genius_lyrics)

moosehead_lyrics = DatasetDescriptor(
    "moosehead_lyrics",
    ["artist", "song", "link", "text"],
    "song",
    "artist",
    "text",
    None, )
moosehead_lyrics_streamer = CSVDatasetStreamer(moosehead_lyrics)

_movies_meta = DatasetDescriptor(
    "movies_meta",
    ["adult", "belongs_to_collection", "budget", "genres", "homepage", "id", "imdb_id", "original_language", "original_title", "overview", "popularity", "poster_path", "production_companies", "production_countries", "release_date",
     "revenue", "runtime", "spoken_languages", "status", "tagline", "title", "video", "vote_average", "vote_count"],
    "title",
    "production_companies",
    "imdb_id",
    "genres", )
_movies_subtitles = DatasetDescriptor(
    "movies_subtitles",
    ["start_time", "end_time", "text", "imdb_id"],
    "imdb_id", None, "text", None, )


class MovieDatasetStreamer(CSVDatasetStreamer):
    def __init__(self, force_rebuild=False):
        super().__init__(None)

        self.dataset_path = os.path.join('temp', 'movies_dataset.db')
        if not os.path.exists('temp'):  os.makedirs('temp')

        build_db = not os.path.exists(self.dataset_path) or force_rebuild

        self.connection = sqlite3.connect(self.dataset_path)
        if build_db:
            cursor = self.connection.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS movies_meta (imdb_id INTEGER PRIMARY KEY, title TEXT, artist TEXT, genre TEXT)")
            cursor.execute("CREATE TABLE IF NOT EXISTS movies_subtitle (xid INTEGER PRIMARY KEY, start_time DECIMAL, text TEXT, imdb_id INTEGER REFERENCES movies_meta(imdb_id))")

            with _movies_meta.open() as file:
                reader = csv.reader(file)
                indices = _movies_meta.get_indices(next(reader))
                for line in reader:
                    title, production_companies, imdb_id, genres = [line[i] for i in indices]
                    if imdb_id == '': continue
                    production_companies = ",".join([company['name'] for company in json5.loads(production_companies)])
                    genres = ",".join(sorted([genre['name'] for genre in json5.loads(genres)]))
                    cursor.execute("INSERT OR IGNORE INTO movies_meta VALUES (?, ?, ?, ?)", (imdb_id[2:], title, production_companies, genres))
                    self.connection.commit()

            with _movies_subtitles.open() as file:
                reader = csv.reader(file)
                next(reader)
                for line in reader:
                    start_time, _, text, imdb_id = line
                    if imdb_id == '': continue
                    cursor.execute("INSERT OR IGNORE INTO movies_subtitle (start_time, text, imdb_id) VALUES (?, ?, ?)", (start_time, text, imdb_id[:2]))
                    self.connection.commit()

            cursor.close()

    def stream(self) -> Iterator[list[str]]:
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM movies_meta")
        cursor.close()
