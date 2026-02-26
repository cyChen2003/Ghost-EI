# -*- coding: utf-8 -*-

import subprocess
import os
import time
import copy
import torch
import shutil
import concurrent.futures
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Callable, Optional, Sequence
import base64
import openai
from pathlib import Path
import yaml
from PIL import Image, ImageDraw
import openai
from MobileAgent.foreground_hook import ForegroundAppHook, send_broadcast_for_overlay, send_emulator_sms
# === External dependencies (kept consistent with original script) ===
from MobileAgent.api import inference_chat
from MobileAgent.text_localization import ocr
from MobileAgent.icon_localization import det
from MobileAgent.controller import get_screenshot, tap, slide, type, back, home, take_screenshot
from MobileAgent.prompt import (
    get_action_prompt,
    get_reflect_prompt,
    get_memory_prompt,
    get_process_prompt,
    get_judge_prompt,
)
from MobileAgent.chat import (
    init_action_chat,
    init_reflect_chat,
    init_memory_chat,
    init_judge_chat,
    add_response,
    add_response_two_image,
)
from MobileAgent.judge_client import JudgeClient, copy_judgement_contents
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dashscope import MultiModalConversation
import dashscope
from MobileAgent.android_env import AndroidEnv, EnvConfig
from MobileAgent.mobile_agent import MobileAgent
import web.page1
import json


def build_overlay_trigger(
    *,
    adb_path: str,
    component: str,
    action: str,
    title: str,
    content: str,
    cancel: str,
    confirm: str,
    weburl: str,
    is_urgent: bool = True,
) -> Callable[[], None]:
    def _trigger() -> None:
        print("[hook] App detected in foreground, sending broadcast...")
        send_broadcast_for_overlay(
            adb_path=adb_path,
            component=component,
            action=action,
            title=title,
            content=content,
            cancel=cancel,
            confirm=confirm,
            weburl=weburl,
            is_urgent=is_urgent,
        )
        time.sleep(2)
    return _trigger

def build_popup_trigger(*, adb_path: str, phone_numbers: Sequence[str], content: str) -> Callable[[], None]:
    numbers = [str(p).strip() for p in phone_numbers if str(p).strip()]
    message = (content or "").strip()

    def _trigger() -> None:
        if not numbers or not message:
            print("[hook] Popup SMS configuration missing numbers or message; skipping injection.")
            return
        print("[hook] App detected in foreground, injecting inbound SMS via emulator console...")
        for phone in numbers:
            send_emulator_sms(adb_path=adb_path, phone=phone, content=message)
            time.sleep(0.5)
    return _trigger


CONFIG_ENV_VAR = "GHOST_EI_CONFIG"
DEFAULT_CONFIG_PATH = "./config.yaml"


def load_run_configuration(config_path: str) -> Tuple[EnvConfig, Dict[str, Any], Dict[str, Any]]:
    """Load YAML configuration and split into env/runtime/hook sections."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        raw_config: Dict[str, Any] = yaml.safe_load(f) or {}

    env_section = raw_config.get("env")
    if not env_section:
        raise KeyError("Configuration file missing required 'env' section.")

    env_cfg = EnvConfig(**env_section)
    runtime_section = raw_config.get("runtime", {})
    hooks_section = raw_config.get("hooks", {})
    return env_cfg, runtime_section, hooks_section


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load scenario dataset from jsonl file."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_output_artifacts(output_dir: str) -> None:
    """Ensure runtime directories exist."""
    os.makedirs(output_dir, exist_ok=True)


def _normalize_app_whitelist(apps: Any) -> List[str]:
    if isinstance(apps, list):
        return apps
    if not apps:
        return []
    return [str(apps)]


def start_overlay_attack_flow(
    *,
    cfg: EnvConfig,
    scenario: Dict[str, Any],
    overlay_cfg: Dict[str, Any],
) -> Tuple[Optional[ForegroundAppHook], Optional[Any]]:
    extra = scenario.get("extra") or {}
    overlay_title = extra.get("title") or overlay_cfg.get("default_title") or "Security Alert"
    overlay_content = extra.get("content") or overlay_cfg.get("default_content") or "Suspicious activity detected. Please verify."
    overlay_cancel = extra.get("cancel") or overlay_cfg.get("default_cancel") or "Cancel"
    overlay_confirm = extra.get("confirm") or overlay_cfg.get("default_confirm") or "Verify"
    is_urgent = bool(extra.get("is_urgent", True))

    use_page = bool(extra.get("use_page"))
    page_info = extra.get("page_information") or {}
    httpd = None
    weburl = extra.get("weburl") or overlay_cfg.get("server_hostport", "")

    if use_page:
        page_title = page_info.get("title") or overlay_title
        page_fields = page_info.get("form") or []
        httpd, _ = web.page1.start_server_in_thread(page_title, page_fields)
        if not weburl:
            print("[hook] Overlay use_page enabled but no server_hostport configured; web url will be empty.")
    else:
        weburl = ""

    trigger = build_overlay_trigger(
        adb_path=cfg.adb_path,
        component=overlay_cfg.get("component", "com.example.myapplication/.AdbReceiver"),
        action=overlay_cfg.get("action", "com.example.broadcast.UPDATE_POPUP"),
        title=overlay_title,
        content=overlay_content,
        cancel=overlay_cancel,
        confirm=overlay_confirm,
        weburl=weburl,
        is_urgent=is_urgent,
    )

    hook = ForegroundAppHook(
        adb_path=cfg.adb_path,
        apps=_normalize_app_whitelist(scenario.get("app")),
    )
    hook.start(trigger)
    return hook, httpd


def start_popup_sms_flow(
    *,
    cfg: EnvConfig,
    scenario: Dict[str, Any],
) -> Optional[ForegroundAppHook]:
    extra = scenario.get("extra") or {}
    numbers = extra.get("numbers") or []
    message = extra.get("message") or ""
    numbers = [str(n).strip() for n in numbers if str(n).strip()]
    if not numbers or not message:
        print(f"[hook] Popup SMS skipped for scenario {scenario.get('id')}: missing numbers or message.")
        return None

    trigger = build_popup_trigger(
        adb_path=cfg.adb_path,
        phone_numbers=numbers,
        content=message,
    )

    hook = ForegroundAppHook(
        adb_path=cfg.adb_path,
        apps=_normalize_app_whitelist(scenario.get("app")),
    )
    hook.start(trigger)
    return hook

def main() -> None:
    config_path = os.getenv(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)
    cfg, runtime_cfg, hooks_cfg = load_run_configuration(config_path)
    cfg.API_url = cfg.API_url or os.getenv(
        "OPENAI_API_URL",
        "https://api.openai.com/v1/chat/completions",
    )
    cfg.token = cfg.token or os.getenv("OPENAI_API_KEY", "")

    dataset_path = runtime_cfg.get("dataset_path", "./datasets/scenarios_final_v6.jsonl")
    output_dir = runtime_cfg.get("output_dir") or f"{cfg.caption_model}-{cfg.reflection_switch}"
    thinking_output_path = runtime_cfg.get("thinking_output_path") or f"{output_dir}-thinking.jsonl"
    judgement_dir = runtime_cfg.get("judgement_dir", "judgement")
    cfg.judgement_dir = judgement_dir

    overlay_cfg = hooks_cfg.get("overlay", {})

    ensure_output_artifacts(output_dir)

    runtime_env = AndroidEnv(cfg)
    dataset = load_dataset(dataset_path)
    judge_model = JudgeClient(
        model=cfg.judge_model,
        api_key=cfg.token,
        api_url=cfg.API_url,
    )
    counter = 0
    hook: Optional[ForegroundAppHook] = None

    for data in dataset:
        counter += 1
        httpd = None
        judge_model.clear(judgement_dir)
        runtime_env.reset_env()

        hook = None
        use_dynamic = (data.get("use_dynamic") or "").strip()
        if use_dynamic == "Overlay Attack":
            hook, httpd = start_overlay_attack_flow(cfg=cfg, scenario=data, overlay_cfg=overlay_cfg)
        elif use_dynamic == "Popup SMS":
            hook = start_popup_sms_flow(cfg=cfg, scenario=data)

        if data.get("need_env") is True and data.get("files_name"):
            runtime_env.upload_files(
                local_dir=cfg.file_dir,
                files_name=data["files_name"],
            )
            time.sleep(1)
            runtime_env.open_files()

        instruction = data["prompt"]

        agent = MobileAgent(runtime_env, instruction)
        agent.run()
        res = judge_model.judge(
            instruction=instruction,
            action_history=agent.output_action_history,
            judge_criteria=data["result"],
            images=judgement_dir,
        )
        print(res)

        with open(thinking_output_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": data["id"],
                        "prompt": instruction,
                        "judge": res["parsed"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        scenario_id = data["id"]
        copy_judgement_contents(
            dest_dir=os.path.join(output_dir, str(scenario_id)),
            src_dir=judgement_dir,
        )
        judge_model.clear(judgement_dir)

        if hook:
            hook.stop()
            hook = None
        if httpd:
            httpd.shutdown()
            httpd.server_close()
        runtime_env.zero_background_via_ui()


if __name__ == "__main__":
    main()
