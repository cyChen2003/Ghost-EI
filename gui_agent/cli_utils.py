from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def _json_dict(value: str) -> Dict[str, Any]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON for remote-extra-params: {exc}") from exc
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("remote-extra-params must decode to a JSON object.")
    return data


def _normalize_optional_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_extra_params(value: Any) -> Optional[Dict[str, Any]]:
    if value in (None, "", {}):
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"remote_extra_params must be valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("remote_extra_params JSON must decode to an object.")
        return parsed
    raise ValueError("remote_extra_params must be a dict or JSON-encoded dict.")


def _load_inference_defaults(config_path: Optional[str]) -> Dict[str, Any]:
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        return {
            "inference_mode": "local",
            "remote_api_url": None,
            "remote_api_key": None,
            "remote_model": None,
            "remote_temperature": None,
            "remote_top_p": None,
            "remote_extra_params": None,
            "remote_timeout": 120.0,
        }

    with path.open("r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh) or {}
    inference_cfg = raw_cfg.get("inference") or {}

    defaults = {
        "inference_mode": str(inference_cfg.get("mode", "local")).strip().lower(),
        "remote_api_url": _normalize_optional_str(inference_cfg.get("remote_api_url")),
        "remote_api_key": _normalize_optional_str(inference_cfg.get("remote_api_key")),
        "remote_model": _normalize_optional_str(inference_cfg.get("remote_model")),
        "remote_temperature": inference_cfg.get("remote_temperature"),
        "remote_top_p": inference_cfg.get("remote_top_p"),
        "remote_extra_params": None,
        "remote_timeout": inference_cfg.get("remote_timeout", 120.0),

    }

    extra_params = inference_cfg.get("remote_extra_params")
    if extra_params is not None:
        defaults["remote_extra_params"] = _coerce_extra_params(extra_params)

    mode = defaults["inference_mode"]
    if mode not in {"local", "remote"}:
        raise ValueError(f"Invalid inference mode configured in {path}: {mode}")

    if defaults["remote_timeout"] is None:
        defaults["remote_timeout"] = 120.0

    return defaults


def add_remote_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to gui_agent config.yaml for inference defaults.",
    )
    parser.add_argument(
        "--inference-mode",
        choices=("local", "remote"),
        default=None,
        help="Choose between local HuggingFace execution or remote vLLM HTTP mode (defaults to config).",
    )
    parser.add_argument(
        "--remote-api-url",
        default=None,
        help="OpenAI-compatible base URL or full /chat/completions endpoint for the remote vLLM server (overrides config).",
    )
    parser.add_argument("--remote-api-key", default=None, help="Optional bearer token for the remote server.")
    parser.add_argument("--remote-basic-user", default=None, help="Username for HTTP Basic auth when the remote server requires it.")
    parser.add_argument("--remote-basic-password", default=None, help="Password for HTTP Basic auth when the remote server requires it.")
    parser.add_argument("--remote-model", default=None, help="Model identifier exposed by the remote vLLM server.")
    parser.add_argument(
        "--remote-temperature",
        type=float,
        default=None,
        help="Override sampling temperature for remote inference (leave empty to use config/server default).",
    )
    parser.add_argument(
        "--remote-top-p",
        type=float,
        default=None,
        help="Override nucleus sampling for remote inference (leave empty to use config/server default).",
    )
    parser.add_argument(
        "--remote-extra-params",
        type=_json_dict,
        default=None,
        help="Optional JSON object with extra parameters forwarded to the remote completion request.",
    )
    parser.add_argument(
        "--remote-timeout",
        type=float,
        default=None,
        help="HTTP timeout (seconds) for each remote vLLM request (defaults to config).",
    )


def extract_remote_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = _load_inference_defaults(args.config_path)
    overrides = {
        "inference_mode": args.inference_mode,
        "remote_api_url": args.remote_api_url,
        "remote_model": args.remote_model,
        "remote_api_key": args.remote_api_key,
        "remote_temperature": args.remote_temperature,
        "remote_top_p": args.remote_top_p,
        "remote_extra_params": args.remote_extra_params,
        "remote_timeout": args.remote_timeout,

    }

    for key, value in overrides.items():
        if value is not None:
            defaults[key] = value

    return defaults
