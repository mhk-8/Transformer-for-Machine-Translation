"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""
import os
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

import sacrebleu
import wandb
from tqdm import tqdm
from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, SOS_IDX, EOS_IDX, PAD_IDX
from lr_scheduler import NoamScheduler

# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        # Convert logits to log-probabilities — KLDivLoss expects log-input
        log_probs = F.log_softmax(logits, dim=-1)   # [N, vocab_size]
 
        # Build the smoothed target distribution from scratch.
        # Start by filling every position with the uniform smooth mass.
        smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
 
        # Place the high-confidence mass on the correct token
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
 
        # The <pad> token should have zero target probability everywhere
        smooth_dist[:, self.pad_idx] = 0.0
 
        # Zero out entire rows where the target itself is <pad> —
        # we don't want to compute loss at padding positions at all
        pad_positions = (target == self.pad_idx)
        smooth_dist[pad_positions] = 0.0
 
        # KL divergence: sum(target * (log_target - log_pred))
        # With reduction='sum' we get the total loss; we'll normalise below.
        loss = F.kl_div(log_probs, smooth_dist, reduction='sum')
 
        # Normalise by the number of real (non-pad) tokens in the batch
        num_tokens = (~pad_positions).sum().float().clamp(min=1.0)
        return loss / num_tokens.clamp(min=1.0)


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train() if is_train else model.eval()
 
    total_loss   = 0.0
    total_tokens = 0
    mode_label   = "Train" if is_train else "Val"
    total_grad_norm = 0.0
    num_batches     = 0
    
    progress = tqdm(
        data_iter,
        desc=f"Epoch {epoch_num} [{mode_label}]",
        leave=False,
        dynamic_ncols=True,
    )
 
    context = torch.enable_grad() if is_train else torch.no_grad()
 
    with context:
        for src, tgt in progress:
            src = src.to(device)   # [B, src_len]
            tgt = tgt.to(device)   # [B, tgt_len]
 
            # ── Teacher forcing split ─────────────────────────────────
            # tgt_in  feeds the decoder:  <sos> w1 w2 ... w_{n-1}
            # tgt_out is what we predict: w1   w2 ... w_{n-1} <eos>
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
 
            # ── Build masks ───────────────────────────────────────────
            # pad_idx=1 matches PAD_IDX in dataset.py and the default
            # in model.py's make_src_mask / make_tgt_mask
            src_mask = make_src_mask(src, pad_idx=PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=PAD_IDX)
 
            # ── Forward pass ──────────────────────────────────────────
            logits = model(src, tgt_in, src_mask, tgt_mask)
            # logits: [B, tgt_len-1, vocab_size]
 
            # Flatten batch and sequence dims for the loss function
            B, T, V = logits.shape
            loss = loss_fn(
                logits.reshape(B * T, V),      # [B*T, vocab_size]
                tgt_out.reshape(B * T),         # [B*T]
            )
 
            # ── Backward pass (training only) ─────────────────────────
            if is_train:
                optimizer.zero_grad()
                loss.backward()
 
                # Gradient clipping — prevents exploding gradients,
                # especially important early in training
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                total_grad_norm += grad_norm.item()
                
                optimizer.step()
                scheduler.step()
 
            # ── Accumulate stats ──────────────────────────────────────
            # Count real (non-padding) tokens for a meaningful average
            num_tokens  = (tgt_out != PAD_IDX).sum().item()
            total_loss  += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches  += 1
            
            current_lr = optimizer.param_groups[0]['lr'] if is_train else 0.0
            progress.set_postfix(loss=f"{loss.item():.4f}",lr=f"{current_lr:.2e}")
 
    avg_loss = total_loss / max(total_tokens, 1)
    avg_grad_norm = total_grad_norm / max(num_batches, 1) if is_train else 0.0
    
    # Log to W&B — only if a run is active
    if wandb.run is not None:
        log_dict = {
            f"{mode_label.lower()}/loss": avg_loss,
            f"{mode_label.lower()}/perplexity": math.exp(min(avg_loss, 20)),
            "epoch": epoch_num,
        }
        if is_train and optimizer is not None:
            log_dict["train/lr"] = optimizer.param_groups[0]['lr']
            log_dict["train/grad_norm"] = avg_grad_norm
        wandb.log(log_dict)
 
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    raw = model.module if hasattr(model, 'module') else model
    raw.eval()
 
    with torch.no_grad():
        # Encode the source sentence once — reused at every decode step
        memory = raw.encode(src, src_mask)   # [1, src_len, d_model]
 
        # Start the output sequence with just <sos>
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
 
        for _ in range(max_len - 1):
            tgt_mask   = make_tgt_mask(ys, pad_idx=PAD_IDX)
            logits = raw.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break
 
    return ys   # [1, out_len]


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════


def _tokens_to_sentence(indices: list[int], tgt_vocab: list[str]) -> str:
    """
    Convert a list of token indices to a clean string, skipping all
    special tokens (<unk>, <pad>, <sos>, <eos>).
 
    Args:
        indices   : Token index list from greedy_decode output.
        tgt_vocab : list[str] — index → token string  (train_ds.tgt_vocab)
 
    Returns:
        Space-joined string of real tokens.
    """
    # Indices 0-3 are the four special tokens — skip them all
    skip = {SOS_IDX, EOS_IDX, PAD_IDX, 0}   # 0 = UNK_IDX
    tokens = [
        tgt_vocab[i]
        for i in indices
        if i not in skip and i < len(tgt_vocab)
    ]
    return ' '.join(tokens)


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab: list,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    model.eval()
 
    hypotheses: list[str] = []   # model's translations
    references: list[str] = []   # gold translations
 
    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval", leave=False):
            # test_dataloader should have batch_size=1 for clean sentence-level decode
            # but we handle batch_size > 1 too (decode each sentence separately)
            for i in range(src.size(0)):
                src_i = src[i].unsqueeze(0).to(device)   # [1, src_len]
                tgt_i = tgt[i].unsqueeze(0).to(device)   # [1, tgt_len]
 
                src_mask = make_src_mask(src_i, pad_idx=PAD_IDX)
 
                # Greedy decode
                output = greedy_decode(
                    model, src_i, src_mask,
                    max_len=max_len,
                    start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )
 
                # Convert indices → strings, stripping all special tokens
                hyp = _tokens_to_sentence(output[0].tolist(), tgt_vocab)
                ref = _tokens_to_sentence(tgt_i[0].tolist(),  tgt_vocab)
 
                hypotheses.append(hyp)
                references.append(ref)
 
    # ── Compute corpus BLEU ───────────────────────────────────────────
    try:
        import sacrebleu
        # sacrebleu expects references as a list of lists (one list per ref)
        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        return bleu.score   # float, 0–100
 
    except ImportError:
        # Fallback: simple unigram precision (rough, but avoids extra deps)
        # For a real submission, install sacrebleu:  pip install sacrebleu
        print("[BLEU] sacrebleu not found — using simple unigram overlap.")
        return _simple_bleu(hypotheses, references)
    
def _simple_bleu(hypotheses: list[str], references: list[str]) -> float:
    """
    Very rough corpus-level 1-gram BLEU fallback.
    Use only if sacrebleu is unavailable — prefer sacrebleu for submissions.
    """
    from collections import Counter
    match = total = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = hyp.split()
        ref_tokens = set(ref.split())
        match += sum(1 for t in hyp_tokens if t in ref_tokens)
        total += len(hyp_tokens)
    if total == 0:
        return 0.0
    precision = match / total
    return round(precision * 100, 2)    
    



# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    raw = model.module if hasattr(model, 'module') else model
 
    enc0 = raw.encoder.layers[0]

    model_config = {
        'src_vocab_size': raw.src_embed.num_embeddings,
        'tgt_vocab_size': raw.tgt_embed.num_embeddings,
        'd_model':        raw.d_model,
        'N':              len(raw.encoder.layers),
        'num_heads':      enc0.self_attn.num_heads,
        'd_ff':           enc0.ffn.linear1.out_features,
        'dropout':        getattr(enc0.dropout, 'p', 0.1),
    }
 
    torch.save(
        {
            'epoch':                epoch,
            'model_state_dict':     raw.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'model_config':         model_config,
        },
        path
    )
    print(f"[checkpoint] Saved epoch {epoch} → {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    checkpoint = torch.load(path, map_location='cpu')
 
    raw = model.module if hasattr(model, 'module') else model
    raw.load_state_dict(checkpoint['model_state_dict'])
 
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
 
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
 
    epoch = checkpoint.get('epoch', 0)
    print(f"[checkpoint] Loaded epoch {epoch} from {path}")
    return epoch

# ══════════════════════════════════════════════════════════════════════
#  W&B SAMPLE TRANSLATION LOGGING
# ══════════════════════════════════════════════════════════════════════
 
def log_sample_translations(
    model:      Transformer,
    dataset,
    num_samples: int = 5,
    device:      str = 'cpu',
    epoch:       int = 0,
) -> None:
    """
    Decode a handful of examples and log them to W&B as a Table.
    This makes it easy to eyeball translation quality during training.
 
    Args:
        model       : Trained Transformer.
        dataset     : Multi30kDataset with data, src_vocab, tgt_vocab populated.
        num_samples : Number of examples to decode (default 5).
        device      : 'cpu' or 'cuda'.
        epoch       : Current epoch, used as the W&B step key.
    """
    if wandb.run is None:
        return
 
    model.eval()
    table = wandb.Table(columns=["epoch", "source (DE)", "reference (EN)", "hypothesis (EN)"])
 
    # Sample from the first num_samples examples for reproducibility
    indices = range(min(num_samples, len(dataset)))
 
    with torch.no_grad():
        for idx in indices:
            src_tensor, tgt_tensor = dataset[idx]
            src_tensor = src_tensor.unsqueeze(0).to(device)
            tgt_tensor = tgt_tensor.unsqueeze(0).to(device)
 
            src_mask = make_src_mask(src_tensor, pad_idx=PAD_IDX)
            output   = greedy_decode(
                model, src_tensor, src_mask,
                max_len=100,
                start_symbol=SOS_IDX,
                end_symbol=EOS_IDX,
                device=device,
            )
 
            # Reconstruct source German sentence
            src_ids  = src_tensor[0].tolist()
            src_str  = _tokens_to_sentence(src_ids,         dataset.src_vocab)
            ref_str  = _tokens_to_sentence(tgt_tensor[0].tolist(), dataset.tgt_vocab)
            hyp_str  = _tokens_to_sentence(output[0].tolist(),     dataset.tgt_vocab)
 
            table.add_data(epoch, src_str, ref_str, hyp_str)
 
    wandb.log({"translations": table, "epoch": epoch})
    
    
# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    config = {
        # Model architecture — matches the paper exactly
        'd_model':      256,
        'N':            3,
        'num_heads':    8,
        'd_ff':         512,
        'dropout':      0.1,
 
        # Training
        'batch_size':   128,
        'num_epochs':   250,
        'warmup_steps': 4000,
        'label_smooth': 0.1,
        'grad_clip':    1.0,
 
        # Data
        'min_freq':     1,
        'max_len':      100,
 
        # Checkpoint
        'ckpt_every':   50,     # save a checkpoint every N epochs
        'ckpt_path':    'checkpoint.pt',
    }
 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[train] Using device: {device}")
 
    # ── 1. Init W&B ───────────────────────────────────────────────────
    wandb.init(
        project="da6401-a3",
        config=config,
        name=f"transformer-d{config['d_model']}-N{config['N']}-h{config['num_heads']}-ff{config['d_ff']}"
    )
 
    # ── 2. Build datasets ─────────────────────────────────────────────
    print("[train] Loading and preprocessing data ...")
 
    train_ds = Multi30kDataset(split='train',min_freq=config['min_freq'])
    train_ds.build_vocab()
    train_ds.process_data()
 
    # Val and test must reuse the training vocab — indices must be identical
    val_ds = Multi30kDataset(split='validation')
    val_ds.src_vocab = train_ds.src_vocab
    val_ds.tgt_vocab = train_ds.tgt_vocab
    val_ds.src_stoi  = train_ds.src_stoi
    val_ds.tgt_stoi  = train_ds.tgt_stoi
    val_ds.process_data()
 
    test_ds = Multi30kDataset(split='test')
    test_ds.src_vocab = train_ds.src_vocab
    test_ds.tgt_vocab = train_ds.tgt_vocab
    test_ds.src_stoi  = train_ds.src_stoi
    test_ds.tgt_stoi  = train_ds.tgt_stoi
    test_ds.process_data()
 
    src_vocab_size = len(train_ds.src_vocab)
    tgt_vocab_size = len(train_ds.tgt_vocab)
    print(f"[train] Vocab sizes — DE: {src_vocab_size:,}  EN: {tgt_vocab_size:,}")
    wandb.config.update({
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size
    })
    
    # ── 3. DataLoaders ────────────────────────────────────────────────
    # num_workers=2 is safe on Kaggle; set to 0 if you hit pickling errors
    train_loader = train_ds.get_dataloader(
        batch_size=config['batch_size'], shuffle=True,  num_workers=2,
    )
    val_loader = val_ds.get_dataloader(
        batch_size=config['batch_size'], shuffle=False, num_workers=2,
    )
    # BLEU eval decodes one sentence at a time — batch_size=1 avoids
    # padding artefacts affecting translation quality measurements
    test_loader = test_ds.get_dataloader(
        batch_size=1, shuffle=False, num_workers=2,
    )
 
    # ── 4. Model ──────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['d_model'],
        N=config['N'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        checkpoint_path=None
    ).to(device)
    
    raw_model = model
    if torch.cuda.device_count() > 1:
        print(f"[train] {torch.cuda.device_count()} GPUs detected — using DataParallel")
        model = torch.nn.DataParallel(model)
 
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model parameters: {param_count:,}")
    wandb.config.update({'param_count': param_count})
 
    # ── 5. Optimizer ──────────────────────────────────────────────────
    # lr=1.0 is intentional — the Noam scheduler scales it down to the
    # correct magnitude (peak ~0.0007 for d_model=512, warmup=4000)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
 
    # ── 6. LR Scheduler ───────────────────────────────────────────────
    scheduler = NoamScheduler(
        optimizer,
        d_model=config['d_model'],
        warmup_steps=config['warmup_steps'],
    )
 
    # ── 7. Loss function ──────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=PAD_IDX,           # = 1, matches dataset.py and model.py
        smoothing=config['label_smooth'],
    )
 
    # ── 8. Training loop ──────────────────────────────────────────────
    print(f"\n[train] Starting training — {config['num_epochs']} epochs\n")
    best_val_loss = float('inf')
 
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
 
        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
        )
 
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            optimizer=None, scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )
 
        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{config['num_epochs']} | "
            f"train={train_loss:.4f}  val={val_loss:.4f} | "
            f"ppl={math.exp(min(val_loss, 20)):.2f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.0f}s"
        )
        
        # Log sample translations every 25 epochs to track qualitative progress
        if epoch % 25 == 0:
            log_sample_translations(
                model, val_ds, num_samples=5, device=device, epoch=epoch,
            )
        # Save best model separately so we can restore it for BLEU eval
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(raw_model, optimizer, scheduler, epoch, path='best_model.pt')
            print(f"  ↳ new best val_loss={val_loss:.4f} — saved best_model.pt")
            
        # Periodic checkpoint so we can resume if Kaggle session restarts
        if epoch % config['ckpt_every'] == 0:
            save_checkpoint(
                raw_model, optimizer, scheduler, epoch,
                path=f"checkpoint_ep{epoch:03d}.pt",
            )
 
    # Always save the final state
    save_checkpoint(
        raw_model, optimizer, scheduler, config['num_epochs'],
        path=config['ckpt_path']
    )
 
    # ── 9. Final BLEU on test set ─────────────────────────────────────
    # Reload best model weights for evaluation
    print("\n[train] Evaluating best model on test set ...")
    load_checkpoint('best_model.pt', raw_model)
    raw_model.to(device)
 
    bleu = evaluate_bleu(
        raw_model,
        test_loader,
        tgt_vocab=train_ds.tgt_vocab,
        device=device,
        max_len=config['max_len'],
    )
 
    print(f"\n[train] Test BLEU: {bleu:.2f}")
    wandb.log({
        'test/bleu':      bleu,
        'best_val_loss':  best_val_loss,
        'best_val_ppl':   math.exp(min(best_val_loss, 20)),
    })
 
    # Log a final batch of sample translations on the test set
    log_sample_translations(
        model, test_ds, num_samples=10, device=device, epoch=config['num_epochs'],
    )
 
    wandb.finish()
    print("[train] Done.")
 
    
    


if __name__ == "__main__":
    run_training_experiment()
