"""
Decoder-only Transformer language model for the Andrews-Curtis conjecture.

Architecture (from the paper):
    n_layers       = 8
    d_model        = 512
    n_heads        = 4    (head_dim = 128)
    context_length = 1024
    vocab_size     = 6    (x, x^{-1}, y, y^{-1}, SEP, EOS)
    tied_embeddings: token embedding matrix W_E is shared with the output
                     unembedding layer W_U (i.e. W_E = W_U^T).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal (autoregressive) self-attention."""

    def __init__(self, d_model: int, n_heads: int, context_length: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Lower-triangular causal mask registered as a non-parameter buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(context_length, context_length)).view(
                1, 1, context_length, context_length
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class MLP(nn.Module):
    """Position-wise feed-forward network (4x expansion, GELU activation)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm → Attention → residual,
    then LayerNorm → MLP → residual."""

    def __init__(self, d_model: int, n_heads: int, context_length: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_length)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ACTransformer(nn.Module):
    """
    Autoregressive decoder-only Transformer for the AC conjecture.

    Default hyper-parameters match the paper:
        n_layers=8, d_model=512, n_heads=4, context_length=1024, vocab_size=6

    Tied embeddings: ``self.unembed.weight`` is the same tensor object as
    ``self.token_emb.weight``, so W_E = W_U^T throughout training.
    """

    def __init__(
        self,
        vocab_size: int = 6,
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 8,
        context_length: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, context_length) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # Output unembedding — weight tied to token embedding (W_E = W_U^T)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.unembed.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        n_layers = len(self.blocks)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for block in self.blocks:
            nn.init.normal_(block.attn.qkv.weight, std=0.02)
            # Scale residual projections by 1/sqrt(2*n_layers) per GPT-2 paper
            nn.init.normal_(block.attn.out_proj.weight, std=0.02 / math.sqrt(2 * n_layers))
            nn.init.normal_(block.mlp.fc1.weight, std=0.02)
            nn.init.normal_(block.mlp.fc2.weight, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Next-token prediction forward pass.

        Parameters
        ----------
        idx : (B, T) long tensor of token IDs

        Returns
        -------
        logits : (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.context_length, (
            f"Sequence length {T} exceeds context length {self.context_length}"
        )
        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.unembed(x)

    def get_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Return the final-layer hidden states for all token positions
        (after the last LayerNorm, before unembedding).

        Used by extract_embeddings.py to read the EOS-token representation.

        Parameters
        ----------
        idx : (B, T) long tensor of token IDs

        Returns
        -------
        hidden : (B, T, d_model)
        """
        B, T = idx.shape
        assert T <= self.context_length
        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)
