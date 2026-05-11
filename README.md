# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

================================================================================
  "Attention Is All You Need" (Vaswani et al., 2017)
  German → English Translation using Multi30k
================================================================================

--------------------------------------------------------------------------------
OVERVIEW
--------------------------------------------------------------------------------

This project is a from-scratch PyTorch implementation of the Transformer architecture described in the paper "Attention Is All You Need" (Vaswani et al., 2017 — https://arxiv.org/abs/1706.03762).

The task is Neural Machine Translation (NMT): translating German sentences into
English using the Multi30k dataset. Every component — attention, positional encoding, the encoder/decoder stacks, the Noam learning rate schedule, and label smoothing — is implemented manually using low-level PyTorch primitives. No high-level transformer libraries are used.

Training is tracked end-to-end with Weights & Biases (W&B), logging loss, perplexity, gradient norms, learning rate, and sample translations at every epoch. Final evaluation is done using corpus-level BLEU via sacrebleu.


--------------------------------------------------------------------------------
PAPER REFERENCE
--------------------------------------------------------------------------------

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).
Attention Is All You Need.
Advances in Neural Information Processing Systems (NeurIPS).
https://arxiv.org/abs/1706.03762


--------------------------------------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------------------------------------

assignment3/
├── model.py          Core Transformer architecture
│                       - scaled_dot_product_attention()
│                       - MultiHeadAttention
│                       - PositionalEncoding
│                       - PositionwiseFeedForward
│                       - EncoderLayer, Encoder
│                       - DecoderLayer, Decoder
│                       - Transformer (encode, decode, forward, infer)
│                       - make_src_mask(), make_tgt_mask()
│
├── dataset.py        Data loading and preprocessing
│                       - Multi30kDataset (HuggingFace + spaCy)
│                       - Vocabulary building from token frequencies
│                       - Token → index conversion
│                       - Padded batching via DataLoader
│
├── lr_scheduler.py   Noam learning rate schedule
│                       - NoamScheduler (LRScheduler subclass)
│                       - get_lr_history() for plotting the LR curve
│
├── train.py          Training pipeline and evaluation
│                       - LabelSmoothingLoss
│                       - run_epoch() — train and val loops
│                       - greedy_decode() — autoregressive inference
│                       - evaluate_bleu() — corpus BLEU via sacrebleu
│                       - save_checkpoint() / load_checkpoint()
│                       - log_sample_translations() — W&B translation table
│                       - run_training_experiment() — full experiment entry point
│
├── requirements.txt  Python dependencies
└── README.txt        This file


--------------------------------------------------------------------------------
ARCHITECTURE DETAILS
--------------------------------------------------------------------------------

The implementation follows the "base" Transformer configuration from the paper:

  Component                   Value
  ─────────────────────────   ─────
  d_model (embedding dim)     512
  Number of layers (N)        6
  Attention heads             8
  d_k = d_v per head          64   (= 512 / 8)
  Feed-forward inner dim      2048
  Dropout                     0.1
  Label smoothing (ε)         0.1
  Optimizer                   Adam (β1=0.9, β2=0.98, ε=1e-9)
  LR schedule                 Noam (warmup_steps=4000)
  Batch size                  128
  Epochs                      30

Key design decisions:
  - Post-norm residual connections (Add & Norm after each sub-layer)
  - Sinusoidal positional encodings, not learned
  - Embeddings scaled by sqrt(d_model) before adding positional encoding
  - Xavier uniform initialisation for all weight matrices
  - Separate embedding tables for source and target vocabularies
  - Shared positional encoding table between encoder and decoder
  - Greedy decoding at inference time (no beam search)


--------------------------------------------------------------------------------
DATASET
--------------------------------------------------------------------------------

Multi30k — a multilingual extension of the Flickr30k dataset containing ~29,000 training sentence pairs, 1,014 validation pairs, and 1,000 test pairs.

  Source : German (DE)
  Target : English (EN)

Loaded automatically from HuggingFace Datasets:
  https://huggingface.co/datasets/bentrevett/multi30k

Tokenisation is done with spaCy:
  - German  : de_core_news_sm
  - English : en_core_web_sm

Vocabulary is built from the training split only. Tokens appearing fewer than min_freq=2 times are mapped to <unk> at runtime. The four special tokens (<unk>, <pad>, <sos>, <eos>) occupy indices 0, 1, 2, 3 respectively.

Typical vocabulary sizes after filtering:
  - German  : ~7,800 tokens
  - English : ~5,900 tokens


--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

Python 3.10 or later is required (the code uses built-in list[...] type hints).

Dependencies:
  torch
  numpy
  matplotlib
  scikit-learn
  wandb
  datasets
  spacy
  sacrebleu
  tqdm

Install all dependencies:

  pip install -r requirements.txt

Download the spaCy language models (required before running anything):

  python -m spacy download de_core_news_sm
  python -m spacy download en_core_web_sm

If you are on a GPU machine, install the appropriate CUDA-enabled PyTorch build from https://pytorch.org/get-started/locally/ before running pip install.


--------------------------------------------------------------------------------
WEIGHTS & BIASES SETUP
--------------------------------------------------------------------------------

Training logs to W&B automatically. You need a free account at https://wandb.ai and your API key set up before running training.

Authenticate once:

  wandb login


If you want to run without W&B (dry run / debugging), disable it:

  wandb disabled

All W&B logging calls in the code are guarded with `if wandb.run is not None`
so the training script works correctly even when W&B is disabled.

What gets logged to W&B:
  - train/loss         — token-level average training loss per epoch
  - train/perplexity   — exp(train_loss), epoch-level
  - train/lr           — current learning rate (Noam schedule)
  - train/grad_norm    — average gradient norm per epoch
  - val/loss           — validation loss per epoch
  - val/perplexity     — exp(val_loss), epoch-level
  - translations       — W&B Table with 5 DE→EN sample translations (every 5 epochs)
  - test/bleu          — corpus-level BLEU on the test set (end of training)
  - best_val_loss      — best validation loss achieved
  - best_val_ppl       — perplexity at best validation loss


--------------------------------------------------------------------------------
HOW TO RUN
--------------------------------------------------------------------------------

1. FULL TRAINING EXPERIMENT
   ─────────────────────────
   Runs all 30 epochs, saves checkpoints, evaluates BLEU on the test set,
   and logs everything to W&B.

     python train.py

   Checkpoints saved:
     best_model.pt           — best validation loss (used for BLEU eval)
     checkpoint_ep005.pt     — every 5 epochs
     checkpoint_ep010.pt
     ...
     checkpoint.pt           — final epoch

   Expected runtime:
     GPU (e.g. T4/V100) : ~2–4 hours for 30 epochs
     CPU                : not recommended (extremely slow)

2. VERIFY MODEL SHAPES (no training)
   ────────────────────────────────────
   Runs a forward pass with random data and prints all tensor shapes.
   Useful to confirm the architecture is wired correctly before training.

     python model.py

   Expected output (among others):
     src_mask shape : torch.Size([2, 1, 1, 10])
     tgt_mask shape : torch.Size([2, 1, 8, 8])
     Forward pass output : torch.Size([2, 8, 1200])
     Trainable parameters: ~65,000,000
     Softmax sums to 1 (softmax check): True
     All checks passed!

3. PLOT THE NOAM LR SCHEDULE (no training)
   ─────────────────────────────────────────
   Simulates and plots the learning rate curve for 20,000 steps.
   Useful for understanding the warmup/decay behaviour before training.

     python lr_scheduler.py

4. RESUME TRAINING FROM A CHECKPOINT
   ─────────────────────────────────────
   Load a checkpoint inside run_training_experiment() or use the utilities
   directly in a script:

     from train import load_checkpoint
     from model import Transformer

     model = Transformer(src_vocab_size=7854, tgt_vocab_size=5893, ...)
     epoch = load_checkpoint("checkpoint_ep010.pt", model)
     # model is now restored to epoch 10 weights

5. TRANSLATE A SINGLE SENTENCE
   ─────────────────────────────
   Use Transformer.infer() after loading a trained model:

     import spacy
     import torch
     from model import Transformer
     from train import load_checkpoint

     spacy_de = spacy.load("de_core_news_sm")

     model = Transformer(src_vocab_size=..., tgt_vocab_size=..., ...)
     load_checkpoint("best_model.pt", model)
     model.eval()

     translation = model.infer(
         src_sentence="Ein Mann sitzt auf einer Bank.",
         spacy_de=spacy_de,
         src_stoi=train_ds.src_stoi,
         tgt_itos=train_ds.tgt_vocab,
         device="cpu",
     )
     print(translation)   # → "a man sits on a bench ."


--------------------------------------------------------------------------------
AUTOGRADER CONTRACT
--------------------------------------------------------------------------------

The following function/method signatures must not be changed. The autograder imports and calls them directly.

  model.py
    scaled_dot_product_attention(Q, K, V, mask)  → (Tensor, Tensor)
    MultiHeadAttention.forward(q, k, v, mask)    → Tensor
    PositionalEncoding.forward(x)                → Tensor
    make_src_mask(src, pad_idx)                  → BoolTensor
    make_tgt_mask(tgt, pad_idx)                  → BoolTensor
    Transformer.encode(src, src_mask)            → Tensor
    Transformer.decode(memory, src_m, tgt, tgt_m)→ Tensor

  train.py
    greedy_decode(model, src, src_mask, max_len,
                  start_symbol, end_symbol, device) → Tensor [1, out_len]
    evaluate_bleu(model, test_dataloader,
                  tgt_vocab, device)                → float (0–100)
    save_checkpoint(model, optimizer,
                    scheduler, epoch, path)         → None
    load_checkpoint(path, model,
                    optimizer, scheduler)           → int (epoch)

Checkpoint format (keys in the saved dict):
  'epoch', 'model_state_dict', 'optimizer_state_dict',
  'scheduler_state_dict', 'model_config'

model_config keys:
  'src_vocab_size', 'tgt_vocab_size', 'd_model',
  'N', 'num_heads', 'd_ff', 'dropout'


--------------------------------------------------------------------------------
KNOWN CONSTRAINTS & DESIGN CHOICES
--------------------------------------------------------------------------------

- torch.nn.MultiheadAttention is NOT used anywhere. Multi-head attention is implemented entirely from nn.Linear and the manual scaled_dot_product_attention function, as required by the assignment.

- torchtext is NOT used. Vocabulary is built with plain Python Counter objects and stored as list[str] + dict[str, int].

- HuggingFace Trainer and PyTorch Lightning are NOT used.

- Beam search is not implemented. Greedy decoding is used at inference time.
  Beam search would improve BLEU by ~1–2 points but is not required.

- The Noam scheduler requires the optimizer to be initialised with lr=1.0.
  The scheduler then scales this base lr by the Noam factor. The peak learning rate for d_model=512, warmup=4000 is approximately 7e-4, reached at step 4000.

- Label smoothing is implemented as a KL divergence loss against a smoothed target distribution (not via nn.CrossEntropyLoss label_smoothing argument), to match the paper exactly and handle the <pad> masking correctly.

- NaN guard in scaled_dot_product_attention: when every key in a row is a padding token, softmax over all -inf produces NaN. These are replaced with 0.0 via torch.nan_to_num to prevent NaN propagation through training.


--------------------------------------------------------------------------------
EXPECTED RESULTS
--------------------------------------------------------------------------------

With the default hyperparameters and 30 epochs on Multi30k:

  Metric              Expected range
  ──────────────────  ──────────────
  Test BLEU           32 – 38
  Best val loss       ~1.5 – 1.8
  Best val perplexity ~4.5 – 6.0
  Parameters          ~65 million

Results will vary slightly depending on random seed, GPU hardware, and the exact spaCy tokenisation version installed.


--------------------------------------------------------------------------------
REFERENCES
--------------------------------------------------------------------------------

[1] Vaswani et al. (2017). Attention Is All You Need.
    https://arxiv.org/abs/1706.03762

[2] Multi30k Dataset (HuggingFace).
    https://huggingface.co/datasets/bentrevett/multi30k

[3] The Annotated Transformer — Harvard NLP.
    https://nlp.seas.harvard.edu/2018/04/03/attention.html

[4] sacrebleu — standardised BLEU evaluation.
    https://github.com/mjpost/sacrebleu

[5] spaCy language models.
    https://spacy.io/models


--------------------------------------------------------------------------------