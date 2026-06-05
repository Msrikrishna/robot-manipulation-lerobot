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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_dino")
@dataclass
class ACTDinoConfig(ACTConfig):
    """ACT policy with a frozen DINOv3 ViT vision backbone (instead of ResNet).

    Identical to ACT in every other respect — same CVAE transformer, action chunking,
    and `L1 + KL` objective — but the image encoder is a **frozen DINOv3 ViT** whose
    patch tokens are reshaped into a 2D feature map and fed into ACT's transformer
    encoder via the same 1x1 conv + 2D sinusoidal position embedding path.

    The ResNet-specific inherited fields (`vision_backbone`,
    `pretrained_backbone_weights`, `replace_final_stride_with_dilation`) are unused.

    Notes
    -----
    * VISUAL normalization is IDENTITY: images stay in [0, 1] and the DINOv3 encoder
      applies its own ImageNet normalization internally (same choice as dino_dt).
    * The DINOv3 backbone is gated on the HF Hub — run `huggingface-cli login` and
      accept the license once before first use.
    """

    # Keep images in [0, 1]; the DINOv3 backbone applies ImageNet normalization itself.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # --- DINOv3 vision backbone ---
    dino_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    dino_image_size: int = 224  # square side fed to DINOv3; must be a multiple of the patch size (16)
    freeze_backbone: bool = True

    def __post_init__(self):
        # Call the grandparent (PreTrainedConfig) directly so we skip ACTConfig's
        # ResNet-only `vision_backbone` validation, then re-apply ACT's other checks.
        PreTrainedConfig.__post_init__(self)

        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model "
                f"invocation. Got {self.n_action_steps} for `n_action_steps` and {self.chunk_size} "
                f"for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.dino_image_size % 16 != 0:
            raise ValueError(
                f"`dino_image_size` must be a multiple of 16 (the DINOv3 /16 patch size). "
                f"Got {self.dino_image_size}."
            )
