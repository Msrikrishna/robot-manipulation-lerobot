#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ACT policy with a frozen DINOv3 ViT vision backbone.

This is the stock Action Chunking Transformer (CVAE transformer + action chunking,
`L1 + KL` objective) with **only the image encoder swapped**: instead of a torchvision
ResNet feature map, images are encoded by a frozen DINOv3 ViT whose patch tokens are
reshaped back into a 2D `(B, C, h, w)` feature map. That feature map flows through the
exact same 1x1 conv projection + 2D sinusoidal position embedding path as ACT, so the
entire `ACT.forward` is reused unchanged (this module only overrides the backbone setup).

The original `act/` files are NOT modified — the reusable layers are imported from there.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoModel

from lerobot.policies.act.modeling_act import (
    ACT,
    ACTDecoder,
    ACTEncoder,
    ACTPolicy,
    ACTSinusoidalPositionEmbedding2d,
    ACTTemporalEnsembler,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.act_dino.configuration_act_dino import ACTDinoConfig
from lerobot.policies.pretrained import PreTrainedPolicy

# DINOv3 was pretrained with ImageNet normalization stats.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DINOv3FeatureMap(nn.Module):
    """Frozen DINOv3 ViT that returns a 2D feature map.

    Mimics torchvision's `IntermediateLayerGetter` output — `{"feature_map": (B, C, h, w)}`
    — so the rest of ACT (the 1x1 conv projection and 2D sinusoidal position embedding)
    is reused verbatim. The DINOv3 patch tokens (CLS + register tokens dropped) are
    reshaped from the `(h*w)` sequence back into the `(h, w)` patch grid.
    """

    def __init__(self, model_id: str, image_size: int, freeze: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_id)
        self.freeze = freeze
        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.hidden_size = self.backbone.config.hidden_size
        self.patch_size = getattr(self.backbone.config, "patch_size", 16)
        self.image_size = image_size
        self.grid = image_size // self.patch_size  # tokens per side (h == w)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        # Keep a frozen backbone in eval mode even when the policy is in train mode
        # (so dropout / batchnorm-like stats inside DINOv3 stay fixed).
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # x: (B, 3, H, W) floats in [0, 1] (VISUAL normalization is IDENTITY upstream).
        x = F.interpolate(
            x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        x = (x - self.mean) / self.std
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.backbone(pixel_values=x)
        tokens = out.last_hidden_state  # (B, seq, hidden); seq = CLS + register + patch tokens
        n_patches = self.grid * self.grid
        # Patch tokens are the last n_patches entries (after CLS + register tokens),
        # in row-major order -> reshape to (B, hidden, grid, grid).
        patches = tokens[:, -n_patches:, :]
        fmap = patches.transpose(1, 2).reshape(x.shape[0], self.hidden_size, self.grid, self.grid)
        return {"feature_map": fmap}


class ACTDino(ACT):
    """ACT network with the ResNet backbone replaced by a frozen DINOv3 ViT.

    Overrides only `__init__` (to build the DINOv3 backbone and size the image
    projection to the DINOv3 hidden dim); `forward` is inherited from `ACT` unchanged
    because `DINOv3FeatureMap` returns the same `{"feature_map": ...}` interface.
    """

    def __init__(self, config: ACTDinoConfig):
        nn.Module.__init__(self)  # bypass ACT.__init__ (which would build a ResNet)
        self.config = config

        # --- VAE encoder (identical to ACT) ---
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0], config.dim_model
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # --- Vision backbone: frozen DINOv3 ViT (the only real change vs. ACT) ---
        if self.config.image_features:
            self.backbone = DINOv3FeatureMap(
                config.dino_model, config.dino_image_size, config.freeze_backbone
            )

        # --- Transformer encoder/decoder (identical to ACT) ---
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            # Project the DINOv3 hidden dim -> dim_model with a 1x1 conv over the patch grid.
            self.encoder_img_feat_input_proj = nn.Conv2d(
                self.backbone.hidden_size, config.dim_model, kernel_size=1
            )
        n_1d_tokens = 1  # latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    # forward() is inherited from ACT unchanged.


class ACTDinoPolicy(ACTPolicy):
    """ACT policy with a frozen DINOv3 backbone.

    Inherits all of `ACTPolicy`'s rollout (`select_action`, action queue, temporal
    ensembling) and training (`forward` -> `L1 + KL`) logic; only the underlying
    network is swapped for `ACTDino`.
    """

    config_class = ACTDinoConfig
    name = "act_dino"

    def __init__(self, config: ACTDinoConfig):
        # Bypass ACTPolicy.__init__ (which builds `ACT`) and replicate it with `ACTDino`.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        self.model = ACTDino(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        self.reset()

    def get_optim_params(self) -> list:
        # The DINOv3 backbone is frozen (requires_grad=False), so simply hand the
        # optimizer every trainable parameter. (ACTPolicy splits out a separate
        # backbone LR group, which would be empty here.)
        return [p for p in self.parameters() if p.requires_grad]
