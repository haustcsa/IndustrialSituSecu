"""
Discrete Denoising Diffusion Probabilistic Model for Protocol Message Generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class DiscreteDDPM(nn.Module):
    """
    Discrete DDPM for generating syntactically correct protocol messages
    """

    def __init__(
            self,
            vocab_size: int,
            max_seq_length: int,
            embedding_dim: int = 256,
            hidden_dim: int = 512,
            num_layers: int = 6,
            num_heads: int = 8,
            timesteps: int = 1000,
            beta_schedule: str = "cosine"
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.timesteps = timesteps

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Transformer backbone for denoising
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

        # Noise schedule
        self.register_buffer("betas", self._get_beta_schedule(beta_schedule, timesteps))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - self.alphas_cumprod))

    def _get_beta_schedule(self, schedule: str, timesteps: int) -> torch.Tensor:
        """Generate noise schedule"""
        if schedule == "linear":
            return torch.linspace(1e-4, 0.
            02, timesteps)
            elif schedule == "cosine":
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.
            0001, 0.9999)
            else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process:  q(x_t | x_0)
        Add noise to discrete tokens using categorical distribution
        """
        batch_size, seq_len = x_0.shape

        # Get noise schedule values for timestep t
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        # Convert tokens to one-hot
        x_0_onehot = F.one_hot(x_0, num_classes=self.vocab_size).float()

        # Add noise:  interpolate between data and uniform distribution
        uniform_prob = torch.ones_like(x_0_onehot) / self.vocab_size
        noisy_prob = (sqrt_alpha_cumprod_t.unsqueeze(-1) * x_0_onehot +
                      sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1) * uniform_prob)

        # Sample from categorical distribution
        x_t = torch.multinomial(
            noisy_prob.view(-1, self.vocab_size),
            num_samples=1
        ).view(batch_size, seq_len)

        return x_t, noisy_prob

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Denoise network: predict x_0 from x_t
        """
        batch_size, seq_len = x.shape

        # Embeddings
        token_emb = self.token_embedding(x)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)
        time_emb = self.time_embedding(t.float().view(-1, 1) / self.timesteps)

        # Combine embeddings
        h = token_emb + pos_emb + time_emb.unsqueeze(1)

        # Transform
        h = self.transformer(h)

        # Predict original distribution
        logits = self.output_projection(h)

        return logits

    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Reverse diffusion step: p(x_{t-1} | x_t)
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)

        # Predict x_0 distribution
        logits = self(x_t, t_tensor)

        if t > 0:
            # Add noise for non-final steps
            probs = F.softmax(logits, dim=-1)
            # Sample with temperature
            temperature = max(0.5, t / self.timesteps)
            x_t_minus_1 = torch.multinomial(
                (probs / temperature).view(-1, self.vocab_size),
                num_samples=1
            ).view(x_t.shape)
        else:
            # Final step: take argmax
            x_t_minus_1 = torch.argmax(logits, dim=-1)

        return x_t_minus_1

    @torch.no_grad()
    def sample(self, batch_size: int, device: str = "cuda") -> torch.Tensor:
        """
        Generate samples via reverse diffusion
        """
        # Start from random tokens
        x_t = torch.randint(
            0, self.vocab_size,
            (batch_size, self.max_seq_length),
            device=device
        )

        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            x_t = self.p_sample(x_t, t)

        return x_t

    def training_step(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Training step: compute denoising loss
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_0.device)

        # Add noise
        x_t, _ = self.q_sample(x_0, t)

        # Predict original
        logits = self(x_t, t)

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            x_0.view(-1)
        )

        return loss


class ProtocolMessageTokenizer:
    """
    Tokenizer for protocol messages
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.byte_to_token = {i: i for i in range(256)}
        self.token_to_byte = {i: i for i in range(256)}

    def encode(self, message: bytes) -> List[int]:
        """Convert message bytes to tokens"""
        return [self.byte_to_token[b] for b in message]

    def decode(self, tokens: List[int]) -> bytes:
        """Convert tokens back to bytes"""
        return bytes([self.token_to_byte[t] for t in tokens])

    def batch_encode(self, messages: List[bytes], max_length: int) -> torch.Tensor:
        """Batch encode with padding"""
        encoded = []
        for msg in messages:
            tokens = self.encode(msg)
            # Pad or truncate
            if len(tokens) < max_length:
                tokens += [0] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
            encoded.append(tokens)
        return torch.tensor(encoded, dtype=torch.long)

    def batch_decode(self, token_tensor: torch.Tensor) -> List[bytes]:
        """Batch decode"""
        return [self.decode(tokens.tolist()) for tokens in token_tensor]