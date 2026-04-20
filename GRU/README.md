# BiGRU Lyrics Generation System

## Project Structure
```
bigru_lyrics/
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters & paths
├── data/
│   └── download_instructions.md
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py         # Text cleaning & normalization
│   │   ├── annotator.py       # Genre + TF-IDF theme keyword extraction
│   │   └── tokenizer.py       # SentencePiece tokenizer training & usage
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── lyrics_dataset.py  # PyTorch Dataset + DataLoader
│   ├── model/
│   │   ├── __init__.py
│   │   └── bigru.py           # BiGRU model architecture
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py         # Training loop, teacher forcing, BLEU eval
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py       # BERTScore, MAUVE, Perplexity, Self-BLEU
│   └── inference/
│       ├── __init__.py
│       └── generator.py       # Inference: start phrase + genre → lyrics
├── preprocess.py              # Entry point: preprocess
├── train.py                   # Entry point: train → evaluate
└── generate.py                # Entry point: load model → generate lyrics
```

## Setup

```bash
pip install -r requirements.txt
```

## Download Dataset
Download the Genius Song Lyrics dataset from:
https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information

Place the CSV file at: `data/song_lyrics.csv`


## Step 1 — Preprocess: Run ONCE (takes ~10 mins). Saves everything to data/processed/
python preprocess.py --data_path data/song_lyrics.csv

## Step 2 — Train: Run as many times as you want, instantly loads cached files
python train.py --output_dir checkpoints/ --epochs 10

## Resume from a checkpoint (optimizer + scheduler state restored too)
python train.py --output_dir checkpoints/ --epochs 10 --resume checkpoints/epoch_005.pt

## Step 3 — Generate lyrics
```bash
python generate.py --checkpoint checkpoints/best_model.pt --start_phrase "i walk alone in" --genre "rock" --num_stanzas 3