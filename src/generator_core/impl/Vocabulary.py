# from ..solution_manager import Vocabulary


# class SimpleVocabulary('Vocabulary'):

#     def __init__(self, special_tokens: set[str], min_freq=2):
#         self.word2idx: dict[str, int] = {"<PAD>": 0, "<UNKNOWN>": 1}
#         self.idx2word: dict[int, str] = {0: "<PAD>", 1: "<UNKNOWN>"}

#         self.word2idx.update({k: i + 2 for i, k in enumerate(special_tokens)})
#         self.idx2word.update({v: k for k, v in self.word2idx.items()})
#         self.word_freq = {}
#         self.min_freq = min_freq

#     def build_vocabulary(self, token: str):
#         self.word_freq[token] = self.word_freq.get(token, 0) + 1
#         if self.word_freq[token] >= self.min_freq and token not in self.word2idx:
#             idx = len(self.word2idx)
#             self.word2idx[token] = idx
#             self.idx2word[idx] = token

#     def encode(self, token: str) -> int:
#         return self.word2idx.get(token, 1)

#     def decode(self, token: int) -> str:
#         return self.idx2word.get(token, '<UNKNOWN>')

#     def __len__(self) -> int:
#         return len(self.word2idx)
