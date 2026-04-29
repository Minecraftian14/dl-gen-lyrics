import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from aspect_midnight import Midnight
from generator_core import *
from .encoder_decoder import EncoderDecoderLSTM, SlidingWindowDataset, SlidingWindowDatasetTruncated

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


class Red(Solution):

    def __init__(self, ds_data: pd.DataFrame):
        """
        Initialize all mid-state helpers
        """

        # For accessing shared tokenizer and embeddings
        self.midnight = Midnight(ds_data, skip_model_loading=True)

        self.ds_data: pd.DataFrame = self.midnight.ds_data
        self.custom_tokens = self.midnight.custom_tokens
        self.genre_to_id = self.midnight.genre_to_id
        self.id_to_genre = self.midnight.id_to_genre
        self.tfidf = self.midnight.tfidf
        self.feature_names = self.midnight.feature_names
        self.vocabulary = self.midnight.vocabulary
        self.embedder = self.midnight.embedder
        self.language_model = self._prepare_language_model()

    def cache_checkpoint_callback_red(self, model_ref, epoch, iteration):
        import os
        import torch
        from generator_core.other_utilities import key_cached
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

        if getattr(model_ref, 'save_cache_periodically', True):
            flight = 'Red._prepare_language_model.cached'
            for file in ['bone', 'pkl']:
                file_path = os.path.join('temp', f'{flight}.{file}')
                if os.path.exists(file_path): os.remove(file_path)
                
            key_cached('cached', lambda: model_ref, group='Red._prepare_language_model')
            print(f"Saved {flight} cache at epoch {epoch}, iteration {iteration}")

        model_ref.eval()
        with torch.no_grad():
            loader = model_ref.trainer.train_dataloader
            for batch_inputs, target_ids in loader:
                padded_anns, windows_x = batch_inputs
                padded_anns = padded_anns.to(next(model_ref.parameters()).device)
                windows_x = windows_x.to(next(model_ref.parameters()).device)
                target_ids = target_ids.to(next(model_ref.parameters()).device)
                
                logits = model_ref(padded_anns, windows_x)
                preds = logits.argmax(dim=-1)
                
                hypotheses_all = []
                references_all = []
                for b in range(preds.size(0)):
                    hypotheses_all.append([str(t.item()) for t in preds[b]])
                    references_all.append([[str(t.item()) for t in target_ids[b]]])
                    
                bleu = corpus_bleu(
                    references_all,
                    hypotheses_all,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=SmoothingFunction().method4,
                )
                print(f"Training BLEU Score (Epoch {epoch}, Iter {iteration}): {bleu:.4f}")
                break
        model_ref.train()

    @cached()
    def _prepare_language_model(self):
        SAVE_CACHE_PERIODICALLY = False
        edm = EncoderDecoderLSTM(
            vocab_size=self.vocabulary.vocab_size(),
            embed_dim=512,
            hidden_dim=512,
            num_layers=2,
            embeddings_weight=self.embedder.embeddings.weight,
        )
        edm.prepare_train(SlidingWindowDatasetTruncated(self))
        edm.save_cache_periodically = SAVE_CACHE_PERIODICALLY
        edm.trainer.on_batch_callback = self.cache_checkpoint_callback_red
        edm.trainer.on_batch_callback_frequency = 50
        return edm

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
        text = re.sub(r"([^\w<>'])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text)
        text = '<SONG_START> ' + text + ' <SONG_END>'
        self.custom_tokens.update({'<SONG_START>', '<SONG_END>'})
        return text

    def pollute_text(self, text: str) -> str:
        text = text.replace(' <NEW_LINE> ', '\n')
        text = re.sub(r" <[\w ']+>", "", text)
        return text

    def _get_top_k_words(self, row, k=5):
        row_data = row.toarray().flatten()
        top_indices = np.argsort(row_data)[-k:]
        return [self.feature_names[i] for i in top_indices if row_data[i] > 0]

    def get_context_words(self, text, k=5):
        return self._get_top_k_words(self.tfidf.transform([text]), k=k)

    def annotate_text(self, id: int, k=5) -> Annotation:
        text = self.ds_data.iloc[id]['lyrics']
        genre = self.ds_data.iloc[id]['tag']
        context_words = self.get_context_words(text, k=k)
        return Annotation(id, genre, context_words)

    def tokenize_text(self, data: str) -> list[int]:
        if isinstance(data, int): data = self.ds_data.iloc[data]['lyrics']
        return self.vocabulary.encode_as_ids(data)

    def detokenize_ids(self, data: list[int]) -> list[str]:
        return self.vocabulary.decode_ids(data)

    @torch.no_grad()
    def embed_tokens(self, data):
        if isinstance(data, int): data = self.ds_data.iloc[data]['lyrics']
        if isinstance(data, str): data = self.tokenize_text(data)
        tokens = torch.tensor(data, dtype=torch.long)
        self.embedder = self.embedder.trainer.to('cpu')
        embeds = self.embedder.embeddings(tokens).numpy()
        return embeds

    @torch.no_grad()
    def inference(self,
                  genre: str, context_words: list[str],
                  starting_words="",
                  starting_token="<SONG_START>", end_token="<SONG_END>",
                  max_len=40, temperature=1.0, top_k=50,
                  ):
        def ends_with(sequence, pattern):
            seq_len, pat_len = sequence.size(1), pattern.size(0)
            if seq_len < pat_len: return False
            return torch.equal(sequence[0, seq_len - pat_len:], pattern)

        def sample_top_k(logits, k=50):
            k = min(k, logits.size(-1))
            top_k_logits, top_k_indices = torch.topk(logits, k)
            probs = F.softmax(top_k_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, num_samples=1)
            return top_k_indices[sampled_idx]

        starting_ids = self.tokenize_text(starting_token + starting_words)
        ending_ids = self.tokenize_text(end_token)
        genre_id = self.tokenize_text(genre)
        context_ids = self.tokenize_text(" ".join(context_words))
        annotation_ids = genre_id + context_ids

        device = next(self.language_model.parameters()).device
        input_ids = torch.tensor(starting_ids, device=device).unsqueeze(0)
        ending_ids = torch.tensor(ending_ids, device=device)
        annotation_ids = torch.tensor([annotation_ids], device=device)

        self.language_model.eval()
        for _ in range(max_len):
            preds = self.language_model(annotation_ids, input_ids)
            preds = preds / temperature
            next_token = sample_top_k(preds.squeeze(0), k=top_k)

            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
            if ends_with(input_ids, ending_ids): break

        input_ids = input_ids.squeeze(0).tolist()
        return self.detokenize_ids(input_ids)

    def inject_sample(self, embeds: np.ndarray, annotation: Annotation) -> 'Sample':
        # We are not using annotations, therefore, we return the embeds as is.
        return embeds

    def prepare_model(self) -> nn.Module:
        return None
        # return M2OLSTM(len(self.vocabulary))

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
