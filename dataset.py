import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
 
import spacy
from datasets import load_dataset

# Special token constants used throughout this file and train.py
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']

class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        self.min_freq = min_freq
 
        # Populated by build_vocab()
        self.src_vocab: list[str]      = []
        self.tgt_vocab: list[str]      = []
        self.src_stoi:  dict[str, int] = {}
        self.tgt_stoi:  dict[str, int] = {}
 
        # Populated by process_data()
        self.data: list[tuple[torch.Tensor, torch.Tensor]] = []
 
        # ── Load raw dataset from HuggingFace Hub ──────────────────────
        print(f"[dataset] Loading Multi30k — split='{split}' ...")
        self.raw = load_dataset("bentrevett/multi30k", split=split)
        print(f"[dataset] {len(self.raw):,} sentence pairs loaded.")
 
        # ── Load spaCy tokenisers ──────────────────────────────────────
        # If this raises OSError, the models haven't been downloaded yet.
        # Fix with:
        #   python -m spacy download de_core_news_sm
        #   python -m spacy download en_core_web_sm
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError as e:
            raise OSError(
                "spaCy language models not found. Run:\n"
                "  python -m spacy download de_core_news_sm\n"
                "  python -m spacy download en_core_web_sm"
            ) from e
 
        # Tokenise every sentence once and cache the token lists.
        # Tokenising inside __getitem__ would re-run spaCy on every
        # single batch access — very slow with multi-worker DataLoaders.
        print(f"[dataset] Tokenising {len(self.raw):,} pairs with spaCy ...")
        self.src_tokens: list[list[str]] = [
            self._tokenise_de(ex['de']) for ex in self.raw
        ]
        self.tgt_tokens: list[list[str]] = [
            self._tokenise_en(ex['en']) for ex in self.raw
        ]
        print("[dataset] Tokenisation complete.")

    # ── Internal tokenisation helpers ─────────────────────────────────
 
    def _tokenise_de(self, text: str) -> list[str]:
        """Lowercase and tokenise a German string using spaCy."""
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]
 
    def _tokenise_en(self, text: str) -> list[str]:
        """Lowercase and tokenise an English string using spaCy."""
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]
    
    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        print(f"[dataset] Building vocabularies (min_freq={self.min_freq}) ...")
 
        # Count every token across all sentences in this split
        src_counts = Counter(tok for sent in self.src_tokens for tok in sent)
        tgt_counts = Counter(tok for sent in self.tgt_tokens for tok in sent)
 
        # Special tokens always occupy indices 0, 1, 2, 3 — then regular
        # tokens sorted alphabetically for deterministic vocab ordering
        self.src_vocab = list(SPECIAL_TOKENS) + sorted(
            tok for tok, freq in src_counts.items() if freq >= self.min_freq
        )
        self.tgt_vocab = list(SPECIAL_TOKENS) + sorted(
            tok for tok, freq in tgt_counts.items() if freq >= self.min_freq
        )
 
        # Build the reverse lookup dicts used during encoding
        self.src_stoi = {tok: idx for idx, tok in enumerate(self.src_vocab)}
        self.tgt_stoi = {tok: idx for idx, tok in enumerate(self.tgt_vocab)}
 
        print(f"[dataset] DE vocab size : {len(self.src_vocab):,} tokens")
        print(f"[dataset] EN vocab size : {len(self.tgt_vocab):,} tokens")

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        if not self.src_stoi or not self.tgt_stoi:
            raise RuntimeError(
                "Vocabulary is empty.  Call build_vocab() first, or assign "
                "src_stoi and tgt_stoi from the training dataset."
            )
 
        print(f"[dataset] Converting tokens to indices ...")
        self.data = []
 
        for src_toks, tgt_toks in zip(self.src_tokens, self.tgt_tokens):
            src_ids = (
                [SOS_IDX]
                + [self.src_stoi.get(tok, UNK_IDX) for tok in src_toks]
                + [EOS_IDX]
            )
            tgt_ids = (
                [SOS_IDX]
                + [self.tgt_stoi.get(tok, UNK_IDX) for tok in tgt_toks]
                + [EOS_IDX]
            )
            self.data.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long),
            ))
 
        print(f"[dataset] {len(self.data):,} examples processed and ready.")
    # ── PyTorch Dataset interface ──────────────────────────────────────
 
    def __len__(self) -> int:
        return len(self.data)
 
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one (src, tgt) LongTensor pair."""
        return self.data[idx]     
    
    
    # ── Collation & DataLoader ─────────────────────────────────────────
 
    def _collate_fn(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a list of variable-length (src, tgt) pairs into two batch tensors.
 
        Sentences in a batch have different lengths — pad_sequence aligns them
        to the longest sentence in that specific batch using PAD_IDX=1.
        batch_first=True gives shape [batch, seq_len] which the Transformer expects.
        """
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
        return src_padded, tgt_padded
 
    def get_dataloader(
        self,
        batch_size:  int  = 128,
        shuffle:     bool = True,
        num_workers: int  = 2,
    ) -> DataLoader:
        """
        Build a DataLoader with automatic per-batch padding.
 
        Args:
            batch_size  : Sentence pairs per batch.
            shuffle     : True for train, False for val/test.
            num_workers : Parallel data-loading workers.
                          Set to 0 when debugging to avoid multiprocessing noise.
 
        Returns:
            DataLoader yielding (src, tgt) LongTensor batches of shape
            [batch_size, max_len_in_that_batch].
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
        )
        
        