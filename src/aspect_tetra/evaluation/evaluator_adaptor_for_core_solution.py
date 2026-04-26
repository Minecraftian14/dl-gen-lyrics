import numpy as np
import torch
from torch import nn

from generator_core import Solution, pad_lists
from .evaluator import compute_bertscore, compute_mauve, compute_self_bleu


class Evaluator:
    def __init__(self, solution: Solution, device=None):
        self.solution = solution
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

    def compute_bertscore(self, n_sample=100, batch_size=10, max_len=500, *extra_args):
        solution = self.solution
        indices = np.random.randint(0, solution.get_data_size(), n_sample)

        scores = {'precision': [], 'recall': [], 'f1': []}
        for i in range(0, n_sample, batch_size):
            indices_batch = indices[i:i + batch_size]
            lyrics = list(map(solution.get_lyrics, indices_batch))
            max_len = min(max_len, max(map(len, lyrics)))
            genres = list(map(solution.get_genre, indices_batch))
            context_words = list(map(" ".join, map(solution.get_context_words, lyrics)))
            generations = solution.bulk_inference(genres=genres, context_words=context_words, max_len=max_len, *extra_args)
            score = compute_bertscore(hypotheses=generations, references=lyrics, device=self.device)
            for key in scores: scores[key].append(score[key])

        return {key: np.mean(scores[key]) for key in scores}

    def compute_mauve(self, n_sample=100, batch_size=10, max_len=500, *extra_args):
        device_id = torch.device(self.device).index
        solution = self.solution

        scores = []
        for _ in range(n_sample // batch_size):
            indices = np.random.randint(0, solution.get_data_size(), batch_size)
            lyrics = list(map(solution.get_lyrics, indices))
            max_len = min(max_len, max(map(len, lyrics)))
            genres = list(map(solution.get_genre, indices))
            context_words = list(map(" ".join, map(solution.get_context_words, lyrics)))
            generations = solution.bulk_inference(genres=genres, context_words=context_words, max_len=max_len, *extra_args)
            # A different distribution for the lyrics
            indices = np.random.randint(0, solution.get_data_size(), batch_size)
            lyrics = list(map(solution.get_lyrics, indices))
            scores.append(compute_mauve(generated_texts=generations, reference_texts=lyrics, device_id=device_id))

        return np.mean(scores)

    def default_collate(self, genres, contexts, lyrics):
        return lyrics

    def default_zip_collate(self, genres, contexts, lyrics):
        return [f"genre {g} {c} {l}" for g, c, l in zip(genres, contexts, lyrics)]

    @torch.no_grad()
    def compute_perplexity(self, n_sample=100, batch_size=10, collate=None, max_len=500, *extra_args):
        if collate is None: collate = self.default_collate

        solution = self.solution
        indices = np.random.randint(0, solution.get_data_size(), n_sample)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        scores = []
        for i in range(0, n_sample, batch_size):
            indices_batch = indices[i:i + batch_size]
            lyrics = list(map(solution.get_lyrics, indices_batch))
            lyrics = [x[:max_len] for x in lyrics]
            genres = list(map(solution.get_genre, indices_batch))
            context_words = list(map(" ".join, map(solution.get_context_words, lyrics)))

            logits = solution.get_logits(list(zip(genres, context_words, lyrics)))
            logits = logits.detach()[:, :-1, :].reshape(-1, logits.shape[-1])

            lyrics = collate(genres, context_words, lyrics)
            lyrics = solution.tokenize_text(lyrics)
            lyrics = pad_lists(lyrics, fill_value=0)
            lyrics = torch.tensor(lyrics, device=self.device)
            lyrics = lyrics[:, 1:].reshape(-1)

            scores.append(torch.exp(criterion(logits, lyrics)).detach().cpu().item())

        return np.mean(scores)

    def compute_self_bleu(self, n_sample=100, batch_size=10, max_len=500, *extra_args):
        solution = self.solution
        indices = np.random.randint(0, solution.get_data_size(), n_sample)

        scores = []
        for i in range(0, n_sample, batch_size):
            indices_batch = indices[i:i + batch_size]
            lyrics = list(map(solution.get_lyrics, indices_batch))
            max_len = min(max_len, max(map(len, lyrics)))
            genres = list(map(solution.get_genre, indices_batch))
            context_words = list(map(" ".join, map(solution.get_context_words, lyrics)))
            generations = solution.bulk_inference(genres=genres, context_words=context_words, max_len=max_len, *extra_args)
            scores.append(compute_self_bleu(generated_texts=generations))

        return np.mean(scores)
