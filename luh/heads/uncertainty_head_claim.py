import torch
import torch.nn as nn
import torch.nn.functional as F

from .uncertainty_head_base import UncertaintyHeadBase

import logging

log = logging.getLogger()


class UncertaintyHeadClaim(UncertaintyHeadBase):
    def __init__(
        self,
        feature_extractor,
        head_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        cfg = None,  # Temporary we save initializing cfg in the head itself
        mask_future_tokens: bool = False,
        claim_chunk_size: int | None = None,
    ):
        super().__init__(feature_extractor, cfg=cfg, model_type="claim")

        self.mask_future_tokens = mask_future_tokens
        self.claim_chunk_size = None if claim_chunk_size is None else int(claim_chunk_size)
        if self.claim_chunk_size is not None and self.claim_chunk_size <= 0:
            raise ValueError("claim_chunk_size must be a positive integer or null")

        self.feature_extractor = feature_extractor
        log.info(f"Feature size: {feature_extractor.feature_dim()}")

        self.proj = nn.Sequential(
                nn.Linear(feature_extractor.feature_dim(), head_dim * 2),
                nn.LayerNorm(head_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim * 2, head_dim),
                nn.LayerNorm(head_dim),
                nn.GELU(),
            )

        #self.position_embedding = nn.Embedding(5000, head_dim)
        self.entity_embedding = nn.Embedding(2, head_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=head_dim, nhead=n_heads, dropout=dropout, activation="gelu", batch_first=True
            )
        # Disable the automatic conversion to NestedTensor because it is not compatible with the
        # src_key_padding_mask we pass (see https://github.com/pytorch/pytorch/issues/113739).
        # Setting `enable_nested_tensor=False` keeps the input as a regular padded tensor and
        # prevents the "MultiheadAttention does not support NestedTensor outside of its fast path"
        # assertion that was raised during evaluation.
        self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers,
                enable_nested_tensor=False,
            )
        
        self.classifier = nn.Sequential(
                nn.Linear(head_dim, head_dim),
                nn.LayerNorm(head_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(head_dim, 1)
            )

        total_params = sum(p.numel() for p in self.parameters())
        log.info(f"Total number of parameters {total_params}")

    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        claims = llm_inputs["claims"]

        # log.debug(f'INFERRING FEATURES OF SHAPE {X.shape}: {X}')
        # log.debug(f'FEATURES ATTENTION MASK: {X_attn_mask.shape}')

        # Don't convert to float32 - maintain original dtype for mixed precision compatibility
        features = X  # Remove .to(torch.float32)
        features = self.proj(features)

        src_key_padding_mask = (X_attn_mask == 0)
        results = []
        batch_size = len(claims)
        #max_tokens = X.size(1)

        for i in range(batch_size):
            entity_mask = claims[i]
            # log.debug(f'USING ENTITY MASK OF SHAPE {entity_mask.shape}: {entity_mask}')

            if len(entity_mask) == 0:
                continue
            claim_chunk_size = self.claim_chunk_size or len(entity_mask)
            claim_results = []
            for start in range(0, len(entity_mask), claim_chunk_size):
                entity_mask_chunk = entity_mask[start:start + claim_chunk_size]
                ent_embeds = self.entity_embedding(entity_mask_chunk)

                out = features[i].unsqueeze(0).expand(ent_embeds.shape[0], -1, -1) + ent_embeds
                src_key_pd = src_key_padding_mask[i].unsqueeze(0).expand(
                    ent_embeds.shape[0], -1
                ).clone()

                assert entity_mask_chunk.shape == src_key_pd.shape
                if self.mask_future_tokens:
                    cumulative_mask = torch.flip(
                        torch.cummax(torch.flip(entity_mask_chunk.int(), dims=[1]), dim=1)[0],
                        dims=[1],
                    ).bool()
                    src_key_pd &= cumulative_mask

                out = self.transformer_encoder(out, src_key_padding_mask=src_key_pd)

                sum_entity_embeds = (out * entity_mask_chunk.unsqueeze(-1)).sum(dim=1)
                count_entity_tokens = entity_mask_chunk.sum(dim=1).clamp(min=1)
                entity_representation = sum_entity_embeds / count_entity_tokens.unsqueeze(-1)
                claim_results.append(self.classifier(entity_representation))

            results.append(torch.cat(claim_results, dim=0))
        
        # Padding to ensure uniform output shape
        max_entities_per_batch = max([o.shape[0] for o in results], default=1)
        padded_results = [F.pad(o, (0, 0, 0, max_entities_per_batch - o.shape[0]), value=-100) for o in results]
        
        return torch.stack(padded_results) if len(padded_results) else torch.zeros(0)

    def forward_from_features(self, features, attention_mask, claims):
        """Run the head on precomputed token features, bypassing the extractor."""
        if features.ndim != 3:
            raise ValueError(f"features must have shape [B, T, H], got {tuple(features.shape)}")
        if attention_mask.shape != features.shape[:2]:
            raise ValueError(
                "attention_mask must match the first two feature dimensions, "
                f"got {tuple(attention_mask.shape)} and {tuple(features.shape)}"
            )
        if len(claims) != features.shape[0]:
            raise ValueError(
                f"claims batch size {len(claims)} does not match features batch size {features.shape[0]}"
            )
        return self._compute_tensors(
            {"claims": claims}, features, attention_mask
        )
