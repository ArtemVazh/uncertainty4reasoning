import gc
import json
import logging
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from argparse import Namespace
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    logging as transformers_logging,
    set_seed,
    AutoConfig
)
from datasets import Dataset, DatasetDict

from luh import AutoUncertaintyHead
from luh.feature_extractors.basic_hidden_states import FeatureExtractorBasicHiddenStates
from luh.feature_extractors.combined import FeatureExtractorCombined
from luh.utils import load_any_dataset

try:
    import inspect
    from speculators.data_generation import vllm_hidden_states_generator as _hs_generator_module
    from speculators.data_generation.vllm_hidden_states_generator import (
        VllmHiddenStatesGenerator as _SpeculatorsVllmHiddenStatesGenerator,
    )
except Exception as e:
    raise ImportError(
        "This script requires speculators.data_generation.vllm_hidden_states_generator.VllmHiddenStatesGenerator"
    ) from e

if "eos_token_id" not in inspect.signature(_hs_generator_module.Request).parameters:
    _VLLM_REQUEST = _hs_generator_module.Request

    def _request_without_legacy_eos(*args, **kwargs):
        kwargs.pop("eos_token_id", None)
        return _VLLM_REQUEST(*args, **kwargs)

    _hs_generator_module.Request = _request_without_legacy_eos


class VllmHiddenStatesGenerator(_SpeculatorsVllmHiddenStatesGenerator):
    """Use local compatibility hooks with the current vLLM request/model APIs."""

    def _create_vllm_config(self, *args, **kwargs):
        original_model_config = _hs_generator_module.ModelConfig
        original_scheduler_config = _hs_generator_module.SchedulerConfig

        def language_model_config(*model_args, **model_kwargs):
            model_kwargs.setdefault("language_model_only", True)
            return original_model_config(*model_args, **model_kwargs)

        def synchronous_scheduler_config(*scheduler_args, **scheduler_kwargs):
            # speculators drives execute/sample/update synchronously itself.
            # vLLM 0.19 otherwise enables AsyncScheduler automatically, which
            # intermittently returns an empty capture for the next request.
            scheduler_kwargs.setdefault("async_scheduling", False)
            return original_scheduler_config(*scheduler_args, **scheduler_kwargs)

        _hs_generator_module.ModelConfig = language_model_config
        _hs_generator_module.SchedulerConfig = synchronous_scheduler_config
        try:
            vllm_config = super()._create_vllm_config(*args, **kwargs)
        finally:
            _hs_generator_module.ModelConfig = original_model_config

            _hs_generator_module.SchedulerConfig = original_scheduler_config
        vllm_config.parallel_config.worker_extension_cls = (
            "train_luh.vllm_hidden_states_worker.HiddenStatesWorkerExtension"
        )
        return vllm_config


transformers_logging.set_verbosity_info()
transformers_logging.enable_default_handler()

log = logging.getLogger(__name__)


def flatten_for_wandb(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}/{k}" if prefix else str(k)
            out.update(flatten_for_wandb(v, new_prefix))
    elif isinstance(obj, (list, tuple)):
        # keep lists out of scalar metric logging
        return {}
    else:
        try:
            out[prefix] = float(obj)
        except Exception:
            pass
    return out


def maybe_init_wandb(config, output_dir: Path):
    if getattr(config, "report_to", None) != "wandb":
        return None

    try:
        import wandb
    except Exception as e:
        raise ImportError("report_to=wandb but wandb is not installed.") from e

    cfg = OmegaConf.to_container(config, resolve=True, throw_on_missing=False)
    project = os.environ.get("WANDB_PROJECT", getattr(getattr(config, "wandb", {}), "project", None))
    name = os.environ.get("WANDB_NAME", getattr(getattr(config, "wandb", {}), "name", None))
    entity = os.environ.get("WANDB_ENTITY", getattr(getattr(config, "wandb", {}), "entity", None))
    group = os.environ.get("WANDB_GROUP", getattr(getattr(config, "wandb", {}), "group", None))
    tags = getattr(getattr(config, "wandb", {}), "tags", None)
    notes = getattr(getattr(config, "wandb", {}), "notes", None)
    mode = getattr(getattr(config, "wandb", {}), "mode", None)

    os.environ.setdefault("WANDB_DIR", str(output_dir))
    run = wandb.init(
        project=project,
        name=name,
        entity=entity,
        group=group,
        tags=list(tags) if tags else None,
        notes=notes,
        mode=mode,
        dir=str(output_dir),
        config=cfg,
    )
    hydra_dir = output_dir / ".hydra"
    if hydra_dir.exists():
        try:
            run.log_artifact(wandb.Artifact("hydra-config", type="config"))
        except Exception:
            pass
    return run


def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.pretrained_model_name_or_path,
        padding_side="left",
        cache_dir=getattr(config, "hf_cache", None),
        token=getattr(config, "hf_token", None),
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def has_valid_uncertainty_supervision(example):
    verified = example["verified"]
    claims = example["claims"]

    def flatten(x):
        if isinstance(x, list):
            for y in x:
                yield from flatten(y)
        else:
            yield x

    verified_flat = list(flatten(verified))
    has_valid_label = any(v != -100 and not (isinstance(v, float) and np.isnan(v)) for v in verified_flat)

    has_nonempty_claim = False
    if claims is not None:
        for claim in claims:
            aligned = claim.get("aligned_token_ids", [])
            if any(int(t) >= 0 for t in aligned):
                has_nonempty_claim = True
                break

    return has_valid_label and has_nonempty_claim


def load_data(config, tokenizer):
    log.info("Loading dataset %s", config.dataset.path)
    dataset = load_any_dataset(config.dataset.path, config)

    if getattr(config.dataset, "label_key_claim", "verified") != "verified":
        dataset = dataset.remove_columns(["verified"])
        dataset = dataset.rename_column(config.dataset.label_key_claim, "verified")

    if getattr(config.dataset, "label_key_token", "uncertainty_labels") != "uncertainty_labels":
        if "uncertainty_labels" in dataset.column_names:
            dataset = dataset.remove_columns(["uncertainty_labels"])
        dataset = dataset.rename_column(config.dataset.label_key_token, "uncertainty_labels")

    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    if getattr(config.dataset, "num_instances", None):
        dataset["train"] = dataset["train"].select(range(config.dataset.num_instances))

    if hasattr(config.dataset, "max_input_length") and config.dataset.max_input_length is not None:
        max_input_length = config.dataset.max_input_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_input_length)

    dataset = dataset.filter(has_valid_uncertainty_supervision)

    tokenized_data = dataset["train"]
    if "test" not in dataset:
        tokenized_data = tokenized_data.train_test_split(
            test_size=config.dataset.test_size,
            seed=42,
        )
    else:
        val_dataset_name = getattr(config.dataset, "validation", "eval")
        tokenized_data = DatasetDict({"train": tokenized_data, "test": dataset[val_dataset_name]})

    if hasattr(config.dataset, "subset") and config.dataset.subset is not None:
        subset_size = config.dataset.subset
        if len(tokenized_data["train"]) > subset_size:
            tokenized_data["train"] = tokenized_data["train"].select(range(subset_size))
        test_subset_size = min(max(1, subset_size // 4), len(tokenized_data["test"]))
        if len(tokenized_data["test"]) > test_subset_size:
            tokenized_data["test"] = tokenized_data["test"].select(range(test_subset_size))

    prompt_template = Path(config.dataset.prompt_path).read_text()

    def prompt_tokens(inst):
        inp = prompt_template.format(q=inst["question"])
        input_ids = tokenizer(inp, return_tensors="pt")["input_ids"][0]
        return {"prompt_tokens": input_ids.tolist()}

    tokenized_data = tokenized_data.map(prompt_tokens)

    def dataset_filter(inst):
        return len(inst["claims"]) > 0

    tokenized_data = tokenized_data.filter(dataset_filter)
    log.info("Train size: %d | Test size: %d", len(tokenized_data["train"]), len(tokenized_data["test"]))
    return tokenized_data


def load_additional_test_datasets(config, tokenizer):
    if not getattr(config, "additional_test_datasets", {}):
        return {}

    prompt_template = Path(config.dataset.prompt_path).read_text()
    additional_datasets = {}

    def prompt_tokens(inst):
        inp = prompt_template.format(q=inst["question"])
        input_ids = tokenizer(inp, return_tensors="pt")["input_ids"][0]
        return {"prompt_tokens": input_ids.tolist()}

    for name, path in config.additional_test_datasets.items():
        log.info("Loading additional test dataset '%s' from %s", name, path)
        dataset = load_any_dataset(path, config)
        if isinstance(dataset, DatasetDict):
            if "test" in dataset:
                dataset_test = dataset["test"]
            else:
                dataset_test = dataset["train"].train_test_split(
                    test_size=config.dataset.test_size,
                    seed=42,
                )["test"]
        else:
            dataset_test = dataset.train_test_split(
                test_size=config.dataset.test_size,
                seed=42,
            )["test"]

        dataset_test = dataset_test.map(prompt_tokens)
        dataset_test = dataset_test.filter(has_valid_uncertainty_supervision)
        dataset_test = dataset_test.filter(lambda inst: len(inst["claims"]) > 0)
        additional_datasets[name] = dataset_test
        log.info("Loaded additional dataset '%s' with %d examples", name, len(dataset_test))

    return additional_datasets


class VLLMStepReasoningCollator:
    """
    Collates raw examples for on-the-fly vLLM hidden-state extraction.

    Returns:
      - input_ids_list: list[list[int]] full prompt+answer token ids, no padding
      - claims: list[Tensor[num_claims_i, T-1]] aligned to features for next-token positions
      - verified: FloatTensor[B, max_claims] padded with -100
    """

    def __init__(self, tokenizer, claim_num_upper_bound: int = -1, sample_claims: bool = False):
        self.tokenizer = tokenizer
        self.claim_num_upper_bound = int(claim_num_upper_bound)
        self.sample_claims = sample_claims

    def _claim_indices(self, num_claims: int) -> list[int]:
        if self.claim_num_upper_bound <= 0 or num_claims <= self.claim_num_upper_bound:
            return list(range(num_claims))
        if self.sample_claims:
            return sorted(random.sample(range(num_claims), self.claim_num_upper_bound))
        return list(range(self.claim_num_upper_bound))

    def _adjust_claim_positions(self, context_length: int, input_ids: list[int], claim_obj: dict) -> torch.Tensor:
        claim_token_positions = claim_obj["aligned_token_ids"]
        mapping = []
        for idx, token_id in enumerate(input_ids[context_length:]):
            if token_id not in self.tokenizer.all_special_ids:
                mapping.append(idx)

        adjusted_positions = [mapping[i] for i in claim_token_positions if i < len(mapping)]
        if len(adjusted_positions) == 0:
            return torch.empty(0, dtype=torch.long)
        return (context_length + torch.tensor(adjusted_positions)).long()

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids_list: list[list[int]] = []
        claims_list: list[torch.Tensor] = []
        verified_rows: list[list[float]] = []

        for e in examples:
            input_ids = list(e["input_ids"])
            prompt_tokens = list(e["prompt_tokens"])
            context_length = len(prompt_tokens)
            full_len = len(input_ids)

            input_ids_list.append(input_ids)

            raw_claims = list(e["claims"])
            raw_verified = list(e["verified"])
            if len(raw_claims) != len(raw_verified):
                raise ValueError(
                    f"claims/verified length mismatch: {len(raw_claims)} != {len(raw_verified)}"
                )
            claim_indices = self._claim_indices(len(raw_claims))

            instance_claims = []
            for claim_idx in claim_indices:
                claim = raw_claims[claim_idx]
                mask = torch.zeros(full_len, dtype=torch.long)
                claim_positions = self._adjust_claim_positions(context_length, input_ids, claim)
                if len(claim_positions) > 0:
                    claim_positions = claim_positions[claim_positions < full_len]
                    mask[claim_positions] = 1
                instance_claims.append(mask[1:])  # drop BOS / first token to match features length T-1

            if len(instance_claims) == 0:
                claim_tensor = torch.zeros((0, max(0, full_len - 1)), dtype=torch.long)
            else:
                claim_tensor = torch.stack(instance_claims, dim=0)
            claims_list.append(claim_tensor)

            verified = [
                -100.0 if (isinstance(raw_verified[i], float) and np.isnan(raw_verified[i]))
                else float(raw_verified[i])
                for i in claim_indices
            ]
            verified_rows.append(verified)

        max_claims = max((len(v) for v in verified_rows), default=0)
        padded_verified = torch.full((len(verified_rows), max_claims), -100.0, dtype=torch.float32)
        for i, row in enumerate(verified_rows):
            if len(row) > 0:
                padded_verified[i, : len(row)] = torch.tensor(row, dtype=torch.float32)

        return {
            "input_ids_list": input_ids_list,
            "claims": claims_list,
            "verified": padded_verified,
        }


class VLLMHiddenStateProvider:
    """
    On-the-fly hidden-state extraction for full gold sequences.

    Important assumptions:
      - uhead feature extractor is hidden-states-only
      - returned features are aligned to next-token positions, shape [B, T-1, H_total]
    """

    def _resolve_vllm_layer_ids(self, uhead) -> list[int]:
        fe = getattr(uhead, "feature_extractor")
        feature_extractors = getattr(fe, "_feature_extractors", [fe]) if isinstance(fe, FeatureExtractorCombined) else [
            fe]

        layer_ids = []
        unsupported = []
        for f in feature_extractors:
            if isinstance(f, FeatureExtractorBasicHiddenStates):
                # Project-specific mapping used in your VLLMUncertaintyHeadFeatures:
                # HF layer i corresponds to layer i-1 in vLLM hidden-state generator
                for x in f._layer_nums:
                    if int(x) >= 0:
                        layer_ids.append(int(x) - 1)
                    else:
                        layer_ids.append(AutoConfig.from_pretrained(self.model_path).num_hidden_layers + x)
            else:
                unsupported.append(f.__class__.__name__)

        if unsupported:
            raise NotImplementedError(
                "This training script currently supports hidden-state-only uheads. "
                f"Unsupported feature extractors: {unsupported}"
            )
        if not layer_ids:
            raise ValueError("Could not infer hidden-state layer ids from uhead.feature_extractor")
        return layer_ids

    def __init__(self, config, uhead, hs_generator=None):
        self.config = config
        self.uhead = uhead

        self.model_path = getattr(config.vllm, "model_path", config.model.pretrained_model_name_or_path)
        self.hs_generator_kwargs = OmegaConf.to_container(
            getattr(config.vllm, "hs_generator_kwargs", {}),
            resolve=True,
        )
        self.layer_ids = self._resolve_vllm_layer_ids(uhead)

        if hs_generator is not None:
            self.hs_generator = hs_generator
        else:
            self.hs_generator = self._build_hs_generator()

        log.info("vLLM hidden-state layers: %s", self.layer_ids)

    def _build_hs_generator(self):
        return VllmHiddenStatesGenerator(
            layer_ids=self.layer_ids,
            model_path=self.model_path,
            **self.hs_generator_kwargs,
        )

    def _reload_hs_generator(self):
        log.warning("Reloading VllmHiddenStatesGenerator")
        try:
            del self.hs_generator
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.hs_generator = self._build_hs_generator()
        return self.hs_generator

    def _reset_hs_prefix_cache(self) -> bool:
        """
        Best-effort reset of prefix cache ONLY for the hidden-state generator path.
        """
        try:
            sch = getattr(self.hs_generator, "scheduler", None)
            if sch is None:
                return False

            if hasattr(sch, "reset_prefix_cache"):
                return bool(sch.reset_prefix_cache())

            kvm = getattr(sch, "kv_cache_manager", None)
            if kvm is not None and hasattr(kvm, "reset_prefix_cache"):
                return bool(kvm.reset_prefix_cache())
        except Exception as e:
            log.warning("Could not reset HS generator prefix cache: %s", e)

        return False

    @staticmethod
    def _payload_seq_len(payload: Optional[dict]) -> int:
        if payload is None or "hidden_states" not in payload or not payload["hidden_states"]:
            return 0

        t0 = torch.as_tensor(payload["hidden_states"][0])
        if t0.ndim == 3 and t0.shape[0] == 1:
            t0 = t0[0]
        if t0.ndim != 2:
            t0 = t0.reshape(-1, t0.shape[-1])
        return int(t0.shape[0])

    def _run_generate(self, input_ids_list: list[list[int]]):
        try:
            return self.hs_generator.generate(input_ids_list)
        except Exception as e:
            log.warning("Hidden-state generation failed: %s", e)
            return None

    def _recover_single_payload(self, seq_ids: list[int]) -> Optional[dict]:
        """
        Recompute one sequence only:
          1) reset HS prefix cache and retry
          2) reload HS generator and retry
        """
        expected = len(seq_ids)

        # Try after prefix-cache reset
        reset_ok = self._reset_hs_prefix_cache()
        if reset_ok:
            log.warning(
                "Retrying hidden-state extraction after HS prefix cache reset "
                "(expected_len=%d).",
                expected,
            )
        else:
            log.warning(
                "HS prefix cache reset unavailable/unsuccessful; retrying anyway "
                "(expected_len=%d).",
                expected,
            )

        payloads = self._run_generate([seq_ids])
        payload = payloads[0] if payloads else None
        got = self._payload_seq_len(payload)
        if got >= expected:
            log.warning(
                "Recovered hidden states after prefix-cache reset: got=%d expected=%d",
                got, expected
            )
            return payload

        log.warning(
            "Skipping sample: still short after prefix-cache reset: got=%d expected=%d",
            got, expected,
        )
        return None

        # Try after full generator reload
        raise Exception(
            ("Still short after prefix-cache reset: got={} expected={}. "
            "Reloading HS generator and retrying.").format(got, expected),
        )
        self._reload_hs_generator()

        payloads = self._run_generate([seq_ids])
        payload = payloads[0] if payloads else None
        got = self._payload_seq_len(payload)
        if got >= expected:
            log.warning(
                "Recovered hidden states after HS generator reload: got=%d expected=%d",
                got, expected
            )
            return payload

        return payload

    @staticmethod
    def _as_2d_float_cpu(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().cpu()
        if t.ndim == 3 and t.shape[0] == 1:
            t = t[0]
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        else:
            t = t.float()
        if t.ndim != 2:
            raise ValueError(f"Expected [T, H] or [1, T, H], got {tuple(t.shape)}")
        return t.contiguous()

    @torch.no_grad()
    def get_features(self, input_ids_list: list[list[int]], device: torch.device):
        payloads = self._run_generate(input_ids_list)
        if payloads is None:
            payloads = [None] * len(input_ids_list)

        per_sample_feats = []
        lengths = []
        feat_dim = None
        keep_indices = []
        skipped_indices = []

        for i, seq_ids in enumerate(input_ids_list):
            expected = len(seq_ids)
            payload = payloads[i] if i < len(payloads) else None
            got = self._payload_seq_len(payload)

            if got < expected:
                log.warning(
                    "Short hidden-state payload for sample %d: got=%d expected=%d. "
                    "Recomputing this sample individually.",
                    i, got, expected,
                )
                payload = self._recover_single_payload(seq_ids)
                got = self._payload_seq_len(payload)

            if payload is None or "hidden_states" not in payload or not payload["hidden_states"]:
                skipped_indices.append(i)
                continue

            if got < expected:
                log.warning(
                    "Skipping sample %d: recovered hidden states still shorter than input "
                    "(got=%d expected=%d)",
                    i, got, expected,
                )
                skipped_indices.append(i)
                continue

            try:
                hs_layers = [self._as_2d_float_cpu(h) for h in payload["hidden_states"]]
                seq_len = hs_layers[0].shape[0]
                for h in hs_layers[1:]:
                    if h.shape[0] != seq_len:
                        raise RuntimeError("Returned hidden-state layers have different sequence lengths")

                sample = torch.cat(hs_layers, dim=-1)

                if sample.shape[0] > expected:
                    sample = sample[:expected]
                if sample.shape[0] < expected:
                    log.warning(
                        "Skipping sample %d after cropping: got=%d expected=%d",
                        i, sample.shape[0], expected,
                    )
                    skipped_indices.append(i)
                    continue

                sample = sample[:-1]  # next-token alignment
            except Exception as e:
                log.warning("Skipping sample %d due to feature construction error: %s", i, e)
                skipped_indices.append(i)
                continue

            per_sample_feats.append(sample)
            lengths.append(sample.shape[0])
            feat_dim = sample.shape[-1]
            keep_indices.append(i)

        if skipped_indices:
            log.warning(
                "Skipped %d/%d samples in batch due to hidden-state extraction failures. "
                "indices=%s",
                len(skipped_indices),
                len(input_ids_list),
                skipped_indices[:20],
            )

        if not per_sample_feats:
            feats = torch.empty((0, 0, 0), dtype=torch.float32, device=device)
            attn_mask = torch.empty((0, 0), dtype=torch.long, device=device)
            return feats, attn_mask, keep_indices, skipped_indices

        max_len = max(lengths)
        batch_size = len(per_sample_feats)
        feats = torch.zeros((batch_size, max_len, feat_dim), dtype=torch.float32)
        attn_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, sample in enumerate(per_sample_feats):
            T = sample.shape[0]
            if T > 0:
                feats[i, :T] = sample
                attn_mask[i, :T] = 1

        return (
            feats.to(device, non_blocking=True),
            attn_mask.to(device, non_blocking=True),
            keep_indices,
            skipped_indices,
        )


class VLLMTrainableUncertaintyModel(nn.Module):
    def __init__(self, uhead, feature_provider, device: torch.device, pos_weight: float = 1.0):
        super().__init__()
        self.uhead = uhead.to(device)
        self.feature_provider = feature_provider
        self.device_ = device
        self.pos_weight = torch.tensor([pos_weight], device=device, dtype=torch.float32)

    def forward_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        features, attention_mask, keep_indices, skipped_indices = self.feature_provider.get_features(
            batch["input_ids_list"], self.device_
        )

        if not keep_indices:
            zero = next(self.uhead.parameters()).sum() * 0.0
            empty = torch.empty((0, 0), device=self.device_, dtype=torch.float32)
            return zero, empty, empty, 0

        claims = [batch["claims"][i].to(self.device_) for i in keep_indices]
        labels = batch["verified"][keep_indices].to(self.device_)

        logits = self.uhead.forward_from_features(
            features=features,
            attention_mask=attention_mask,
            claims=claims,
        ).squeeze(-1)

        if logits.shape != labels.shape:
            raise RuntimeError(f"Logits shape {tuple(logits.shape)} does not match labels shape {tuple(labels.shape)}")

        mask = labels != -100
        if mask.sum() == 0:
            loss = logits.sum() * 0.0
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[mask],
                labels[mask],
                pos_weight=self.pos_weight,
            )

        return loss, logits, labels, len(keep_indices)


@dataclass
class EvalOutput:
    metrics: dict[str, float]
    logits: np.ndarray
    labels: np.ndarray


def compute_binary_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    mask = labels != -100
    labels = labels[mask]
    logits = logits[mask]

    if labels.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
        }

    probas = 1.0 / (1.0 + np.exp(-logits))
    preds = (probas > 0.5).astype(np.int64)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

    unique_labels = np.unique(labels)
    if unique_labels.size >= 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probas))
        precs, recs, _ = precision_recall_curve(labels, probas)
        metrics["pr_auc"] = float(auc(recs, precs))
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    return metrics


@torch.no_grad()
def evaluate_model(model: VLLMTrainableUncertaintyModel, dataloader: DataLoader, desc: str) -> EvalOutput:
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    steps = 0

    for batch in tqdm(dataloader, desc=desc, leave=False):
        loss, logits, labels, num_kept = model.forward_batch(batch)
        if num_kept == 0:
            continue

        total_loss += float(loss.item())
        steps += 1

        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        mask_np = labels_np != -100

        if np.any(mask_np):
            all_logits.append(logits_np[mask_np].astype(np.float32, copy=False))
            all_labels.append(labels_np[mask_np].astype(np.float32, copy=False))

        del loss, logits, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,), dtype=np.float32)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.float32)

    metrics = compute_binary_metrics(logits, labels)
    metrics["loss"] = total_loss / max(steps, 1)
    return EvalOutput(metrics=metrics, logits=logits, labels=labels)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def maybe_reinit_uhead(uhead):
    def reinitialize_weights(module):
        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.ndim >= 2:
                init.xavier_uniform_(module.weight)
            else:
                init.uniform_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            init.zeros_(module.bias)

    uhead.apply(reinitialize_weights)
    return uhead


def build_uhead(config):
    """
    Reuses AutoUncertaintyHead for consistency with your existing project.

    Caveat:
      if AutoUncertaintyHead construction requires a real HF base model to infer feature dims,
      this does a one-time HF model load at startup. The base model is discarded immediately after.
    """
    log.info("Initializing uncertainty head via AutoUncertaintyHead")

    base_model = Namespace(config=AutoConfig.from_pretrained(config.model.pretrained_model_name_or_path))

    try:
        if getattr(config.ue_layer, "path", None):
            uhead = AutoUncertaintyHead.from_pretrained(config.ue_layer.path, base_model)
        else:
            uhead = AutoUncertaintyHead.from_config(config.ue_layer.head_cfg, base_model)
            uhead = maybe_reinit_uhead(uhead)
    finally:
        del base_model
        gc.collect()

    return uhead


def optimizer_to(optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device, non_blocking=True)


def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    checkpoint_dir: Path,
    model: "VLLMTrainableUncertaintyModel",
    optimizer,
    scheduler,
    completed_epochs: int,
    global_step: int,
    best_metric: float,
    no_improve_epochs: int,
    history: list,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "completed_epochs": completed_epochs,   # number of fully finished epochs
        "global_step": global_step,
        "best_metric": best_metric,
        "no_improve_epochs": no_improve_epochs,
        "history": history,
        "uhead_state_dict": model.uhead.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": get_rng_state(),
    }

    tmp_path = checkpoint_dir / "trainer_state.pt.tmp"
    final_path = checkpoint_dir / "trainer_state.pt"
    torch.save(state, tmp_path)
    os.replace(tmp_path, final_path)

    # Optional but useful: save the standalone uhead too
    model.uhead.save(checkpoint_dir / "uhead")

    log.info(
        "Saved resume checkpoint to %s (completed_epochs=%d, global_step=%d)",
        final_path,
        completed_epochs,
        global_step,
    )


def load_training_checkpoint(
    resume_from,
    model: "VLLMTrainableUncertaintyModel",
    optimizer,
    scheduler,
    device: torch.device,
):
    resume_from = Path(resume_from)
    state_path = resume_from / "trainer_state.pt" if resume_from.is_dir() else resume_from

    if not state_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {state_path}")

    state = torch.load(state_path, map_location="cpu")

    model.uhead.load_state_dict(state["uhead_state_dict"], strict=True)

    if state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        optimizer_to(optimizer, device)

    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    set_rng_state(state.get("rng_state"))

    log.info(
        "Loaded checkpoint from %s (completed_epochs=%d, global_step=%d)",
        state_path,
        int(state.get("completed_epochs", 0)),
        int(state.get("global_step", 0)),
    )

    return {
        "completed_epochs": int(state.get("completed_epochs", 0)),
        "global_step": int(state.get("global_step", 0)),
        "best_metric": float(state.get("best_metric", -float("inf"))),
        "no_improve_epochs": int(state.get("no_improve_epochs", 0)),
        "history": list(state.get("history", [])),
    }


def save_metrics(path: Path, metrics: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metrics, f, indent=2)


def build_dataloaders(config, tokenized_data, additional_test_datasets, tokenizer):
    claim_cap = int(getattr(config.ue_layer, "claim_num_upper_bound", -1) or -1)
    train_collator = VLLMStepReasoningCollator(
        tokenizer,
        claim_num_upper_bound=claim_cap,
        sample_claims=True,
    )
    eval_collator = VLLMStepReasoningCollator(
        tokenizer,
        claim_num_upper_bound=claim_cap,
        sample_claims=False,
    )
    log.info("Using claim cap %d for vLLM train/eval batches", claim_cap)
    train_loader = DataLoader(
        tokenized_data["train"],
        batch_size=config.training_arguments.per_device_train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_collator,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        tokenized_data["test"],
        batch_size=getattr(
            config.training_arguments,
            "per_device_eval_batch_size",
            config.training_arguments.per_device_train_batch_size,
        ),
        shuffle=False,
        num_workers=0,
        collate_fn=eval_collator,
        pin_memory=torch.cuda.is_available(),
    )
    extra_eval_loaders = {
        name: DataLoader(
            ds,
            batch_size=getattr(
                config.training_arguments,
                "per_device_eval_batch_size",
                config.training_arguments.per_device_train_batch_size,
            ),
            shuffle=False,
            num_workers=0,
            collate_fn=eval_collator,
            pin_memory=torch.cuda.is_available(),
        )
        for name, ds in additional_test_datasets.items()
    }
    return train_loader, eval_loader, extra_eval_loaders


def mean_eval_metrics(metrics_by_name: dict[str, dict[str, float]]) -> dict[str, float]:
    avg_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    out = {}
    for metric in avg_metrics:
        vals = [m[metric] for m in metrics_by_name.values() if metric in m]
        if vals:
            out[f"mean_{metric}"] = float(np.mean(vals))
    return out


hydra_cfg_path = os.environ.get("HYDRA_CONFIG", None)
hydra_cfg_dir = str(Path(hydra_cfg_path).parent) if hydra_cfg_path is not None else None
hydra_cfg_name = str(Path(hydra_cfg_path).name) if hydra_cfg_path is not None else None


@hydra.main(version_base=None, config_path=hydra_cfg_dir, config_name=hydra_cfg_name)
def main(config):
    import multiprocessing as mp
    import torch.multiprocessing as tmp

    for _mp in (mp, tmp):
        try:
            _mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    checkpoint_dir = Path(getattr(config, "checkpoint_dir", output_dir / "resume_checkpoint"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log.info("Checkpoint directory: %s", checkpoint_dir)

    wandb_run = maybe_init_wandb(config, output_dir)

    seed = int(getattr(config, "seed", 42))
    set_all_seeds(seed)

    train_device = torch.device(getattr(config, "uhead_device", "cuda:0") if torch.cuda.is_available() else "cpu")
    log.info("Training uhead on device: %s", train_device)

    tokenizer = load_tokenizer(config)
    tokenized_data = load_data(config, tokenizer)
    additional_test_datasets = load_additional_test_datasets(config, tokenizer)

    uhead = build_uhead(config)
    log.info("Initializing vLLM hidden-state provider")
    feature_provider = VLLMHiddenStateProvider(config, uhead)
    log.info("vLLM hidden-state provider is ready")
    if os.environ.get("CUDNN_DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info("Moving uncertainty head to %s", train_device)
    model = VLLMTrainableUncertaintyModel(
        uhead=uhead,
        feature_provider=feature_provider,
        device=train_device,
        pos_weight=float(getattr(config.ue_layer, "pos_weight", 1.0)),
    )
    log.info("Uncertainty head is ready on %s", train_device)

    train_loader, eval_loader, extra_eval_loaders = build_dataloaders(
        config,
        tokenized_data,
        additional_test_datasets,
        tokenizer,
    )

    log.info("Creating AdamW optimizer")
    optimizer = AdamW(
        model.uhead.parameters(),
        lr=float(config.training_arguments.learning_rate),
        weight_decay=float(config.training_arguments.weight_decay),
    )
    log.info("AdamW optimizer is ready")

    num_epochs = int(config.training_arguments.num_train_epochs)
    grad_accum = int(getattr(config.training_arguments, "gradient_accumulation_steps", 1))
    total_train_steps = int(np.ceil(len(train_loader) / grad_accum) * num_epochs)
    warmup_ratio = float(getattr(config.training_arguments, "warmup_ratio", 0.0))
    warmup_steps = int(total_train_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    max_grad_norm = float(getattr(config.training_arguments, "max_grad_norm", 1.0))
    best_metric = -float("inf")
    best_ckpt = checkpoint_dir / "best_uhead"  # keep best model in stable resume dir
    early_stopping_patience = int(getattr(config.training_arguments, "early_stopping_patience", 5))
    no_improve_epochs = 0

    history = []
    global_step = 0
    start_epoch = 0

    resume_from = getattr(config, "resume_from_checkpoint", None)
    if resume_from:
        resumed = load_training_checkpoint(
            resume_from=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=train_device,
        )
        start_epoch = resumed["completed_epochs"]
        global_step = resumed["global_step"]
        best_metric = resumed["best_metric"]
        no_improve_epochs = resumed["no_improve_epochs"]
        history = resumed["history"]

        log.info(
            "Resuming training from epoch %d / %d",
            start_epoch + 1 if start_epoch < num_epochs else start_epoch,
            num_epochs,
        )

    if getattr(config, "do_train", True):
        for epoch in range(start_epoch, num_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            num_loss_steps = 0
            accum_counter = 0

            pbar = tqdm(train_loader, desc=f"train epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(pbar, start=1):
                if step == 1:
                    log.info("Starting first hidden-state training batch")
                loss, _, _, num_kept = model.forward_batch(batch)

                if num_kept == 0:
                    pbar.set_postfix(loss=f"{running_loss / max(num_loss_steps, 1):.4f}", kept=0)
                    continue

                loss = loss / grad_accum
                loss.backward()

                running_loss += float(loss.item())
                num_loss_steps += 1
                accum_counter += 1

                if accum_counter % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.uhead.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                pbar.set_postfix(loss=f"{running_loss / max(num_loss_steps, 1):.4f}", kept=num_kept)

            if accum_counter % grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(model.uhead.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            log.info("Saving pre-eval checkpoint for completed epoch %d", epoch + 1)
            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                completed_epochs=epoch + 1,
                global_step=global_step,
                best_metric=best_metric,
                no_improve_epochs=no_improve_epochs,
                history=history,
            )
            batch = None
            loss = None
            _ = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            metrics_by_name = {}
            main_eval = evaluate_model(model, eval_loader, desc="eval")
            metrics_by_name["eval"] = main_eval.metrics

            for name, loader in extra_eval_loaders.items():
                out = evaluate_model(model, loader, desc=f"eval:{name}")
                metrics_by_name[name] = out.metrics

            mean_metrics = mean_eval_metrics(metrics_by_name)
            score = mean_metrics.get("mean_f1", metrics_by_name["eval"]["f1"])
            epoch_summary = {
                "epoch": epoch + 1,
                "train_loss": running_loss / max(len(train_loader), 1),
                "eval": metrics_by_name,
                "eval_mean": mean_metrics,
                "score": score,
            }
            history.append(epoch_summary)
            log.info(json.dumps(epoch_summary, indent=2))
            save_metrics(output_dir / "metrics" / f"epoch_{epoch + 1}.json", epoch_summary)

            if wandb_run is not None:
                wb_metrics = flatten_for_wandb({
                    "train": {"loss": epoch_summary["train_loss"]},
                    "eval": metrics_by_name,
                    "eval_mean": mean_metrics,
                    "score": score,
                    "epoch": epoch + 1,
                    "lr": scheduler.get_last_lr()[0] if scheduler is not None else float("nan"),
                })
                wandb_run.log(wb_metrics, step=global_step)
                wandb_run.summary["best_score_so_far"] = max(float(wandb_run.summary.get("best_score_so_far", -1e18)),
                                                             float(score))

            if score > best_metric:
                best_metric = score
                no_improve_epochs = 0
                best_ckpt.mkdir(parents=True, exist_ok=True)
                model.uhead.save(best_ckpt)
                save_metrics(output_dir / "metrics" / "best_metrics.json", epoch_summary)
                if wandb_run is not None:
                    wandb_run.summary["best_epoch"] = epoch + 1
                    wandb_run.summary["best_score"] = float(score)
                    for k, v in flatten_for_wandb(
                            {"best": {"eval": metrics_by_name, "eval_mean": mean_metrics}}).items():
                        wandb_run.summary[k] = v
                log.info("Saved new best uhead to %s", best_ckpt)
            else:
                no_improve_epochs += 1

            save_training_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                completed_epochs=epoch + 1,
                global_step=global_step,
                best_metric=best_metric,
                no_improve_epochs=no_improve_epochs,
                history=history,
            )

            if getattr(config, "do_save_checkpoints", True) and no_improve_epochs >= early_stopping_patience:
                log.info("Early stopping at epoch %d", epoch + 1)
                break

    if getattr(config, "do_eval", True):
        metrics_by_name = {}
        main_eval = evaluate_model(model, eval_loader, desc="final-eval")
        metrics_by_name["eval"] = main_eval.metrics
        for name, loader in extra_eval_loaders.items():
            out = evaluate_model(model, loader, desc=f"final-eval:{name}")
            metrics_by_name[name] = out.metrics
        summary = {
            "eval": metrics_by_name,
            "eval_mean": mean_eval_metrics(metrics_by_name),
        }
        log.info("Final evaluation:\n%s", json.dumps(summary, indent=2))
        save_metrics(output_dir / "metrics" / "final_eval.json", summary)
        if wandb_run is not None:
            wandb_run.log(flatten_for_wandb({"final_eval": summary}),
                          step=global_step if 'global_step' in locals() else 0)
            for k, v in flatten_for_wandb({"final_eval": summary}).items():
                wandb_run.summary[k] = v

    if getattr(config, "do_save_final_model", True):
        save_path = output_dir / "model"
        model.uhead.save(save_path)
        log.info("Saved final uhead to %s", save_path)
        if getattr(config, "save_dir", None) is not None:
            model.uhead.save(Path(config.save_dir))
            log.info("Saved final uhead to %s", config.save_dir)
        if getattr(config, "hf_save_path", None) is not None:
            model.uhead.push_to_hub(config.hf_save_path)
            log.info("Pushed final uhead to HF: %s", config.hf_save_path)

    save_metrics(output_dir / "metrics" / "history.json", {"history": history})

    if wandb_run is not None:
        try:
            wandb_run.summary["output_dir"] = str(output_dir)
            wandb_run.summary["history_len"] = len(history)
        finally:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
