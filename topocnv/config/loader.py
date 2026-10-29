"""Configuration loading, deep merging, and argparse adaptation utilities."""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from .defaults import get_stage_defaults
from .schema import ExperimentConfig

PathLike = Union[str, os.PathLike]


def load_config(path):
    """读取旧版 v4 的 JSON 配置（也允许 JSON 流式风格的 YAML 文件）。"""
    with open(path) as f:
        return json.load(f)


def save_config(cfg, path):
    """按旧版格式保存配置，并创建父目录。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def load_yaml(path: PathLike) -> Dict[str, Any]:
    """Load a YAML file and return its top-level mapping."""
    try:
        import yaml
    except ImportError as exc:
        message = "PyYAML is required to load YAML configuration files"
        raise ImportError(message) from exc

    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError("YAML configuration must contain a top-level mapping")
    return content


def load_json(path: PathLike) -> Dict[str, Any]:
    """Load a JSON file and return its top-level mapping."""
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        content = json.load(handle)
    if not isinstance(content, dict):
        raise ValueError("JSON configuration must contain a top-level object")
    return content


def merge_configs(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> Dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_configs(current, value)
        else:
            merged[key] = value
    return merged


def _as_bool(value: Any) -> bool:
    """Normalize argparse integer and string switches to booleans."""
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


def _namespace_values(
    args: Union[argparse.Namespace, Mapping[str, Any]]
) -> Dict[str, Any]:
    if isinstance(args, argparse.Namespace):
        return vars(args).copy()
    if isinstance(args, Mapping):
        return dict(args)
    raise TypeError("args must be an argparse.Namespace or mapping")


def _argparse_overrides(
    args: Union[argparse.Namespace, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Translate flat legacy CLI names into the nested configuration schema."""
    values = _namespace_values(args)
    result: Dict[str, Any] = {}

    top_level = {
        "stage",
        "fold",
        "gpu",
        "exp_name",
        "exp_suffix",
        "data_root",
        "results_root",
        "num_folds",
        "seed",
        "stage1_pred_dir",
        "stage1_exp_suffix",
    }
    data_fields = {
        "patch_size",
        "batch_size",
        "num_workers",
        "spacing",
        "spacing_params",
        "clip_lower",
        "clip_upper",
        "normalization",
        "normalization_lower",
        "normalization_upper",
        "ensure_foreground",
        "random_crop_prob",
        "force_cache",
        "use_cache",
        "cache_dir",
    }
    model_fields = {
        "model",
        "backbone",
        "in_channels",
        "num_classes",
        "embed_dim",
        "heads",
        "head_dim",
        "kv_latent_dim",
        "use_mla",
        "use_memory_bank",
        "num_prototypes_per_class",
        "memory_top_k",
        "memory_momentum",
        "memory_temperature",
        "memory_warmup_epochs",
        "memory_type",
        "retrieval_mode",
        "s2_head",
        "binary_y_endpoint",
        "use_confidence_gate",
        "gate_tau",
        "gate_alpha_fixed",
        "refuse_legacy_heads",
        "use_hierarchical",
        "b0_single_stage",
        "zero_stage1_prior",
        "prior_fg_only",
        "prior_scramble",
        "dual_path_stages",
        "residual_conv_stages",
        "lambda_mem",
        "mem_sup_weight",
        "mem_align",
        "bone_norm_thr",
        "evidence_temp",
    }
    train_fields = {
        "epochs",
        "lr",
        "optimizer",
        "weight_decay",
        "layer_decay",
        "scheduler",
        "scheduler_power",
        "warmup_epochs",
        "warmup_start_lr",
        "min_lr",
        "early_stopping",
        "early_stop_patience",
        "gradient_accumulation",
        "gradient_clip_norm",
        "validate_every",
        "full_val_every",
        "sw_early_stop_checks",
        "resume",
        "resume_checkpoint",
        "select_by_sliding_window",
        "prior_dropout_prob",
        "lambda_group",
        "lambda_branch",
        "lambda_aux",
        "proto_loss_weight",
        "proto_init_from_features",
        "proto_weight_start",
        "proto_weight_end",
        "proto_anneal_epochs",
        "loss_bce_weight",
        "loss_dice_weight",
        "focal_alpha",
        "focal_gamma",
        "amp",
    }
    infer_fields = {
        "step_size",
        "overlap",
        "sliding_window_batch_size",
        "sigma_scale",
        "tta",
        "tta_flips",
        "ensemble",
        "skip_sliding_window",
        "input_dir",
        "output_dir",
        "model_path",
        "val_only",
    }
    bool_fields = {
        "force_cache",
        "use_cache",
        "ensure_foreground",
        "use_mla",
        "use_memory_bank",
        "binary_y_endpoint",
        "use_confidence_gate",
        "refuse_legacy_heads",
        "use_hierarchical",
        "b0_single_stage",
        "zero_stage1_prior",
        "prior_fg_only",
        "early_stopping",
        "resume",
        "select_by_sliding_window",
        "proto_init_from_features",
        "amp",
        "tta",
        "ensemble",
        "skip_sliding_window",
        "val_only",
    }

    aliases = {
        "lambda_L": "lambda_l",
        "grad_accum_steps": "gradient_accumulation",
        "memory_patch_grid": "memory_patch_grid",
    }
    for source, target in aliases.items():
        if source in values and values[source] is not None:
            values[target] = values[source]
    model_fields.add("lambda_l")
    model_fields.add("memory_patch_grid")

    for key, value in values.items():
        if value is None:
            continue
        key = aliases.get(key, key)
        if key in bool_fields:
            value = _as_bool(value)
        if key in top_level:
            result[key] = value
        elif key in data_fields:
            result.setdefault("data", {})[key] = value
        elif key in model_fields:
            result.setdefault("model", {})[key] = value
        elif key in train_fields:
            result.setdefault("train", {})[key] = value
        elif key in infer_fields:
            result.setdefault("infer", {})[key] = value

    infer_values = result.get("infer", {})
    if "step_size" in infer_values and "overlap" not in infer_values:
        infer_values["overlap"] = 1.0 - float(infer_values["step_size"])

    spacing_path = result.get("data", {}).get("spacing_params")
    if spacing_path and Path(spacing_path).expanduser().is_file():
        spacing_values = load_json(spacing_path)
        data = result.setdefault("data", {})
        data.setdefault("spacing", spacing_values.get("target_spacing"))
        data.setdefault("clip_lower", spacing_values.get("clip_lower"))
        data.setdefault("clip_upper", spacing_values.get("clip_upper"))

    return result


def from_argparse(
    args: Union[argparse.Namespace, Mapping[str, Any]], stage: str
) -> ExperimentConfig:
    """Convert a legacy flat argparse namespace into an ExperimentConfig."""
    merged = merge_configs(get_stage_defaults(stage), _argparse_overrides(args))
    merged["stage"] = str(stage)
    return ExperimentConfig.from_dict(merged)


def load_experiment_config(
    yaml_path: Optional[PathLike] = None,
    args: Optional[Union[argparse.Namespace, Mapping[str, Any]]] = None,
) -> ExperimentConfig:
    """Load defaults, then YAML, environment roots, and CLI overrides.

    User-supplied precedence is YAML < environment variables < CLI parameters.
    """
    yaml_values = load_yaml(yaml_path) if yaml_path is not None else {}
    cli_values = _namespace_values(args) if args is not None else {}
    stage = str(cli_values.get("stage") or yaml_values.get("stage") or "1")

    merged = merge_configs(get_stage_defaults(stage), yaml_values)

    environment: Dict[str, Any] = {}
    data_root = os.environ.get("CNV_DATA_ROOT")
    results_root = os.environ.get("CNV_RESULTS_ROOT")
    if data_root:
        environment["data_root"] = data_root
    if results_root:
        environment["results_root"] = results_root
        environment["data"] = {
            "cache_dir": str(Path(results_root) / "preprocess_cache")
        }
    merged = merge_configs(merged, environment)

    if args is not None:
        merged = merge_configs(merged, _argparse_overrides(args))
    merged["stage"] = stage
    return ExperimentConfig.from_dict(merged)
