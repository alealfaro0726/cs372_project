
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)

        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attn_output)

        return output


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_output = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(attn_output))

        cross_output = self.cross_attn(x, context, context)
        x = self.norm2(x + self.dropout(cross_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class ConditionalTransformer(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        image_embed_dim: int = 512,
        n_emotions: int = 5,
        emotion_embed_dim: int = 64
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.image_projection = nn.Linear(image_embed_dim, d_model)
        self.emotion_embedding = nn.Embedding(n_emotions, emotion_embed_dim)
        self.emotion_projection = nn.Linear(emotion_embed_dim, d_model)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0

    def forward(
        self,
        tokens: torch.Tensor,
        image_embeds: torch.Tensor,
        emotion_labels: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = tokens.shape
        device = tokens.device

        x = self.token_embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        image_cond = self.image_projection(image_embeds).unsqueeze(1)
        emotion_embeds = self.emotion_embedding(emotion_labels)
        emotion_cond = self.emotion_projection(emotion_embeds).unsqueeze(1)

        context = torch.cat([image_cond, emotion_cond], dim=1)

        causal_mask = self._create_causal_mask(seq_len, device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, context, causal_mask)

        logits = self.output_projection(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=0
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        image_embeds: torch.Tensor,
        emotion_labels: torch.Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        bos_token: int = 1,
        eos_token: int = 2
    ) -> torch.Tensor:
        max_supported_len = self.pos_encoding.pe.size(1)
        if max_length > max_supported_len:
            max_length = max_supported_len  # Prevent positional encoding overflow

        batch_size = image_embeds.size(0)
        device = image_embeds.device

        tokens = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=device)

        for step in range(max_length - 1):
            logits, _ = self.forward(tokens, image_embeds, emotion_labels)

            next_token_logits = logits[:, -1, :] / temperature

            if step > 5:
                for batch_idx in range(batch_size):
                    generated_tokens = tokens[batch_idx, 1:]
                    for token_id in set(generated_tokens.tolist()):
                        count = (generated_tokens == token_id).sum().item()
                        if count > 2:
                            penalty = count ** 1.5
                            next_token_logits[batch_idx, token_id] -= penalty

            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if (next_token == eos_token).all():
                break

        return tokens
