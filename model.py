"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
 
    # Step 1: compute raw attention scores — Q·Kᵀ / √d_k
    # K.transpose(-2, -1) swaps the last two dims: (..., d_k, seq_k)
    # Result: (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
 
    # Step 2: apply mask — fill masked positions with a very large
    # negative number so they become ~0 after softmax
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
 
    # Step 3: softmax over the key dimension to get attention weights
    attn_w = F.softmax(scores, dim=-1)
 
    # Step 4: weighted sum of values
    output = torch.matmul(attn_w, V)
 
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    tgt_len = tgt.size(1)
 
    # Padding mask: True where <pad>, shape [B, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
 
    # Causal mask: upper triangle (excluding diagonal) is True = "future"
    # torch.ones(tgt_len, tgt_len).triu(diagonal=1) gives the upper triangle
    causal_mask = torch.ones(tgt_len, tgt_len, device=tgt.device).triu(diagonal=1).bool()
    # shape: [1, 1, tgt_len, tgt_len] for broadcasting across batch & heads
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
 
    # OR the two: a position is masked if it's padding OR it's in the future
    return pad_mask | causal_mask  # [B, 1, tgt_len, tgt_len]


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        # Four projection matrices: one each for Q, K, V, and the output
        # All map d_model → d_model (the split into heads happens inside forward)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
 
        self.dropout = nn.Dropout(p=dropout)
        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape [batch, seq_len, d_model] → [batch, num_heads, seq_len, d_k]
        so each head operates on its own slice of the embedding.
        """
        B, seq_len, _ = x.size()
        # view splits d_model into (num_heads, d_k), then transpose
        x = x.view(B, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # [B, h, seq_len, d_k]
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Undo _split_heads: [batch, num_heads, seq_len, d_k] → [batch, seq_len, d_model]
        This is the "Concat" operation from the paper.
        """
        B, _, seq_len, _ = x.size()
        # transpose back then merge the head dimension
        x = x.transpose(1, 2).contiguous()  # [B, seq_len, h, d_k]
        return x.view(B, seq_len, self.d_model)  # [B, seq_len, d_model]
    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        # 1. Linear projections
        Q = self._split_heads(self.W_q(query))  # [B, h, seq_q, d_k]
        K = self._split_heads(self.W_k(key))    # [B, h, seq_k, d_k]
        V = self._split_heads(self.W_v(value))  # [B, h, seq_k, d_k]
 
        # 2. Scaled dot-product attention across all heads simultaneously
        # The mask is broadcast over the head dimension automatically
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)  # [B, h, seq_q, d_k]
 
        # 3. Concat heads and project back to d_model
        attn_out = self._merge_heads(attn_out)  # [B, seq_q, d_model]
        return self.W_o(attn_out)               # [B, seq_q, d_model]


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        # Build the positional encoding table: shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
 
        # positions: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
 
        # div_term: [d_model/2] — the 1/10000^(2i/d_model) part
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
 
        # Even indices → sin, odd indices → cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
 
        # Add a batch dimension so it's [1, max_len, d_model]
        # register_buffer: saved in state_dict but not a parameter (won't be updated)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        # self.pe[:, :x.size(1)] slices to the actual sequence length
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        # x: [B, seq_len, d_model]
        # expand → [B, seq_len, d_ff], contract → [B, seq_len, d_model]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════
#  One block of the encoder stack. Two sub-layers:
#    1. Self-attention
#    2. Feed-forward
#  Each wrapped in a residual connection + layer norm (post-norm style, matching the original paper).
class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
 
        # Two layer norms — one per sub-layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        # Sub-layer 1: self-attention (Q = K = V = x, all same sequence)
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
 
        # Sub-layer 2: feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
 
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)  # masked
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)  # encoder–decoder
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
 
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Sub-layer 1: masked self-attention on target sequence
        # Q = K = V = x, but tgt_mask prevents attending to future positions
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
 
        # Sub-layer 2: cross-attention
        # Q comes from decoder (x), K and V come from encoder (memory)
        # src_mask prevents attending to encoder padding
        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
 
        # Sub-layer 3: position-wise feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
 
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        # Create N independent copies of the given layer
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: str = None,
    ) -> None:
        super().__init__()
        # TODO: Instantiate 
        # init should also load the model weights if checkpoint path provided, download the .pth file like this
        super().__init__()
 
        self.d_model = d_model
 
        # --- Embeddings ---
        # Separate embedding tables for source and target vocabularies
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
 
        # Shared positional encoding (same formula for both src & tgt)
        self.pos_enc = PositionalEncoding(d_model, dropout)
 
        # --- Encoder stack ---
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
 
        # --- Decoder stack ---
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(dec_layer, N)
 
        # --- Output projection ---
        # Maps decoder output [*, d_model] → [*, tgt_vocab_size] logits
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
 
        # --- Weight initialisation (Xavier uniform, as in the paper) ---
        self._init_weights()
        
        if checkpoint_path is not None:
            gdown.download(id="<.pth drive id>", output=checkpoint_path, quiet=False)
            self.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for all linear and embedding layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
    
        # Scale embeddings by √d_model (from the paper, §3.4)
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_projection(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)


    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
        self.eval()
 
        # Tokenise and convert to indices
        tokens = [tok.text.lower() for tok in spacy_de.tokenizer(src_sentence)]
        sos_idx = src_vocab.stoi.get('<sos>', 2)
        eos_idx = src_vocab.stoi.get('<eos>', 3)
        unk_idx = src_vocab.stoi.get('<unk>', 0)
 
        src_indices = [sos_idx] + [src_vocab.stoi.get(t, unk_idx) for t in tokens] + [eos_idx]
        src_tensor  = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
 
        with torch.no_grad():
            src_mask = make_src_mask(src_tensor)
            memory   = self.encode(src_tensor, src_mask)
 
            # Start decoding with <sos>
            ys = torch.tensor([[sos_idx]], dtype=torch.long).to(device)
 
            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys       = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == eos_idx:
                    break
 
        # Convert indices back to words, skipping special tokens
        special = {sos_idx, eos_idx, src_vocab.stoi.get('<pad>', 1)}
        tokens_out = []
        for idx in ys[0].tolist():
            if idx in special:
                continue
            tokens_out.append(tgt_vocab.itos[idx] if hasattr(tgt_vocab, 'itos')
                              else tgt_vocab.lookup_token(idx))
 
        return ' '.join(tokens_out)
    
if __name__ == '__main__':
    print("Running shape checks...\n")
 
    B, SRC_LEN, TGT_LEN = 2, 10, 8
    SRC_VOCAB, TGT_VOCAB = 1000, 1200
    D_MODEL, N_HEADS, D_FF, N = 512, 8, 2048, 6
 
    src = torch.randint(2, SRC_VOCAB, (B, SRC_LEN))
    tgt = torch.randint(2, TGT_VOCAB, (B, TGT_LEN))
 
    # Add some padding to verify masks work correctly
    src[0, -2:] = 1  # <pad> index = 1
    tgt[1, -3:] = 1
 
    src_mask = make_src_mask(src)
    tgt_mask = make_tgt_mask(tgt)
 
    print(f"src shape      : {src.shape}")
    print(f"tgt shape      : {tgt.shape}")
    print(f"src_mask shape : {src_mask.shape}  (expected [{B}, 1, 1, {SRC_LEN}])")
    print(f"tgt_mask shape : {tgt_mask.shape}  (expected [{B}, 1, {TGT_LEN}, {TGT_LEN}])")
 
    model = Transformer(SRC_VOCAB, TGT_VOCAB, D_MODEL, N, N_HEADS, D_FF)
    model.eval()
 
    with torch.no_grad():
        logits = model(src, tgt, src_mask, tgt_mask)
 
    print(f"\nForward pass output shape: {logits.shape}  (expected [{B}, {TGT_LEN}, {TGT_VOCAB}])")
 
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {param_count:,}")
 
    # Standalone attention check
    Q = torch.randn(B, N_HEADS, SRC_LEN, D_MODEL // N_HEADS)
    K = torch.randn(B, N_HEADS, SRC_LEN, D_MODEL // N_HEADS)
    V = torch.randn(B, N_HEADS, SRC_LEN, D_MODEL // N_HEADS)
    out, weights = scaled_dot_product_attention(Q, K, V)
    print(f"\nAttention output shape  : {out.shape}")
    print(f"Attention weights shape : {weights.shape}")
    print(f"Weights sum to 1 (softmax check): {weights.sum(dim=-1).allclose(torch.ones(B, N_HEADS, SRC_LEN))}")
 
    print("\nAll checks passed!")