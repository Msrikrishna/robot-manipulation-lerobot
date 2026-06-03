"""
Shared model: frozen DINOv3 (via HF transformers) + a Diffusion Transformer
action head that cross-attends to DINOv3 patch tokens.

Used by BOTH train.py and infer.py so the architectures are guaranteed identical.

Notes
-----
* ViT variants only. The code reads `last_hidden_state` (a token sequence). The
  DINOv3 ConvNeXt variants return feature maps and would need a different path.
* DINOv3 weights on the Hub are gated: accept the license on the model page and
  run `huggingface-cli login` once before first use.
* `transformers` must be recent enough to include DINOv3 (AutoModel routing).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard diffusion timestep embedding. timesteps: [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 backbone -> small trainable linear projection to d_model.

    Returns the full token sequence (CLS + register + patch tokens) so the
    diffusion head can cross-attend over dense spatial features.
    """

    def __init__(self, model_id: str, d_model: int, image_size: int = 256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, d_model)  # trainable
        self.image_size = image_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # keep the backbone frozen + in eval even when the policy is in train mode
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _backbone_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] floats in [0, 1]
        x = F.interpolate(
            x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x)
        return out.last_hidden_state  # [B, seq, hidden]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self._backbone_tokens(x)  # frozen, no grad
        return self.proj(tok)           # [B, seq, d_model]  (grad flows here)


class DiffusionTransformerPolicy(nn.Module):
    """Conditional Diffusion Transformer over action chunks.

    Observation (n_obs_steps frames, multiple cameras, robot state) is encoded
    into a memory token sequence. A transformer-decoder denoiser self-attends
    over the noisy action chunk and cross-attends to that memory, predicting the
    noise (epsilon).
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        cam_keys: list[str],
        n_obs_steps: int,
        horizon: int,
        model_id: str,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        image_size: int = 256,
    ):
        super().__init__()
        self.cam_keys = list(cam_keys)
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.action_dim = action_dim
        self.d_model = d_model

        self.vision = DINOv3Encoder(model_id, d_model, image_size)

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        # tag memory tokens by which camera / which obs step they came from
        self.cam_embed = nn.Parameter(torch.randn(len(self.cam_keys), d_model) * 0.02)
        self.obs_step_embed = nn.Parameter(torch.randn(n_obs_steps, d_model) * 0.02)

        # action chunk tokens
        self.action_in = nn.Linear(action_dim, d_model)
        self.action_pos = nn.Parameter(torch.randn(horizon, d_model) * 0.02)

        # diffusion timestep conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.action_out = nn.Linear(d_model, action_dim)

    def encode_obs(self, batch: dict) -> torch.Tensor:
        """Build the cross-attention memory: [B, M, d_model].

        Expects images as [B, n_obs, 3, H, W] and state as [B, n_obs, state_dim].
        """
        mem = []
        for ci, k in enumerate(self.cam_keys):
            imgs = batch[k]
            for t in range(self.n_obs_steps):
                tok = self.vision(imgs[:, t])  # [B, seq, d]
                tok = tok + self.cam_embed[ci][None, None] + self.obs_step_embed[t][None, None]
                mem.append(tok)

        state = batch["observation.state"]  # [B, n_obs, state_dim]
        for t in range(self.n_obs_steps):
            s = self.state_proj(state[:, t])[:, None]  # [B, 1, d]
            s = s + self.obs_step_embed[t][None, None]
            mem.append(s)

        return torch.cat(mem, dim=1)  # [B, M, d]

    def forward(
        self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise. noisy_actions: [B, horizon, action_dim] -> same shape."""
        x = self.action_in(noisy_actions) + self.action_pos[None]
        temb = self.time_mlp(sinusoidal_embedding(timesteps, self.d_model))  # [B, d]
        x = x + temb[:, None]
        out = self.decoder(tgt=x, memory=memory)
        return self.action_out(out)
