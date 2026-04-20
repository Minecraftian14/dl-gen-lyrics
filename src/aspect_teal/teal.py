import sentencepiece as spm
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.utils.rnn import pad_sequence

from generator_core import *
from aspect_midnight import Word2Vec_SkipGram, ArrayToDatasetForW2V

from .transformer_lm import TransformerModel, TransformerDataset

from dl_trainer import Trainer

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


class Teal(Solution):

    def __init__(self, ds_data: pd.DataFrame):
        """
        Initialize all mid-state helpers
        """

        self.custom_tokens = set()
        self.genre_to_id = dict()

        self.ds_data = self._prepare_ds_data(ds_data)
        # for g in self.ds_data['tag'].unique():
        #     self.custom_tokens.add(f"<GENRE_{g.upper()}>")
        self.custom_tokens = self._get_custom_tokens()
        self.custom_tokens.add('<NEW_LINE>')
        self.genre_to_id = self._get_genre_dict()
        self.id_to_genre = {v: k for k, v in self.genre_to_id.items()}
        self.tfidf = self._prepare_tfidf()
        self.feature_names = self.tfidf.get_feature_names_out()
        
        self.vocabulary = self._prepare_vocabulary()
        self.embedder = self._prepare_embedder()
        self.language_model = self._prepare_language_model()

    @cached()
    def _prepare_ds_data(self, ds_data):
        ds_data = ds_data.copy()
        ds_data['lyrics'] = ds_data['lyrics'].apply(self.clean_text)
        return ds_data

    @cached()
    def _get_custom_tokens(self):
        return self.custom_tokens

    @cached()
    def _get_genre_dict(self):
        return {g: i for i, g in enumerate(self.ds_data['tag'].unique().tolist())}

    @staticmethod
    def tokenize_for_tfidf(text):
        # VERIFY: Specify self.custom_tokens
        # VERIFY: Prevent keywords=['cheek', '?', 'let', ']', '[']
        return [token for token in text.split(' ') if re.match(r"[\w']+", token)]

    @cached()
    def _prepare_tfidf(self):
        tfidf = TfidfVectorizer(
            max_df=0.5,  # ignore very common words
            min_df=5,  # ignore rare words
            stop_words='english',
            max_features=50000,  # limit vocab size TODO: Optimize for this value
            token_pattern=None,
            tokenizer=Teal.tokenize_for_tfidf,
        )
        tfidf.fit(self.ds_data['lyrics'])
        return tfidf

    def _prepare_vocabulary(self):
        # VERIFY: Specify custom_tokens
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
                user_defined_symbols=list(self.custom_tokens)
            )
            os.remove(temp_save)
        else:
            print("Loaded Cache for Midnight._prepare_vocabulary", spm_file)
        sp = spm.SentencePieceProcessor()
        sp.Load(spm_file)
        return sp

    @cached()
    def _prepare_embedder(self):
        word2vec = Word2Vec_SkipGram(
            text_to_ids=self.tokenize_text,
            vocab_size=self.vocabulary.vocab_size(),
            d_embeds=512,
            max_norm=None,
        )
        word2vec.prepare_train(ArrayToDatasetForW2V(self.ds_data['lyrics']))
        return word2vec
    
    
    def model_train_step(self, model, data):
        return model(data[0])
    
    def model_criteria_step(self, criterion, preds, truth):
        preds = preds.permute(0, 2, 1)  # (B, V, T)
        return criterion(preds, truth)

    @cached()
    def _prepare_language_model(self):
        config = {
            "d_model": 512,
            "n_heads": 4,
            "n_layers": 4,
            "ffn": "gelu",
            "norm": "layernorm",
            "attn": "gqa",
            "n_groups": 2,
            "pe": "sinusoidal",
        }
    
        model = TransformerModel(
            vocab_size=self.vocabulary.vocab_size(),
            config=config,
            embedding_weights=self.embedder.embeddings.weight
        )
    
        dataset = TransformerDataset(self)
    
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            collate_fn=dataset.collate_fn
        )
        
        model.trainer = Trainer(
            model=model,
            train_dataloader=dataloader,
            criterion=torch.nn.CrossEntropyLoss(ignore_index=0),
            model_train_step=self.model_train_step,
            model_criteria_step=self.model_criteria_step,
            epochs=1,
            device='cuda',
            record_per_batch_training_loss=True,
            checkpoint_frequency_batch=10,
        )
    
        return model

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

        genre_token = f"<GENRE_{genre.upper()}>"
        cond_text = genre_token + " " + " ".join(context_words)

        starting_ids = self.tokenize_text(cond_text + " " + starting_token + starting_words)
        # starting_ids = self.tokenize_text(starting_token + starting_words)
        ending_ids = self.tokenize_text(end_token)
        # genre_id = self.genre_to_id[genre]
        # context_ids = self.tokenize_text(" ".join(context_words))

        device = next(self.language_model.parameters()).device
        input_ids = torch.tensor(starting_ids, device=device).unsqueeze(0)
        ending_ids = torch.tensor(ending_ids, device=device)
        # context_ids = torch.tensor(context_ids, device=device).unsqueeze(0)
        # genre_id = torch.tensor([genre_id], device=device)

        self.language_model.eval()
        for _ in range(max_len):
            # lengths = torch.tensor([input_ids.size(1)], device=device)

            preds = self.language_model(input_ids)
            preds = preds[:, -1, :] / temperature
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
