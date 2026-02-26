# -*- coding: utf-8 -*-
"""
A lightweight runner that wires Ghost-EI's dynamic hooks (overlay attack and popup SMS)
to the AppAgent pipeline without touching AppAgent internals.

Usage example:
    python run_appagent_hooks.py --dataset ./datasets/scenarios_final_v6.jsonl --scenario-id 1

The script will:
  - load the same configuration used by MobileAgent (config.yaml)
  - start foreground hooks based on each scenario's `use_dynamic` field
  - launch AppAgent's task executor with the scenario prompt and a best-effort doc/device selection
  - auto-evaluate the run via JudgeClient using AppAgent screenshots and step logs
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

from MobileAgent.foreground_hook import ForegroundAppHook, send_broadcast_for_overlay, send_emulator_sms
from MobileAgent.judge_client import JudgeClient
import web.page1

CONFIG_ENV_VAR = "GHOST_EI_CONFIG"
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_DATASET_PATH = "./datasets/scenarios_final_v6.jsonl"
DEFAULT_APPAGENT_ROOT = "AppAgent"


def load_run_configuration(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load YAML configuration and split into env/runtime/hook sections (returned as plain dicts)."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        raw_config: Dict[str, Any] = yaml.safe_load(f) or {}

    env_section = raw_config.get("env")
    if not env_section:
        raise KeyError("Configuration file missing required 'env' section.")

    runtime_section = raw_config.get("runtime", {})
    hooks_section = raw_config.get("hooks", {})
    return env_section, runtime_section, hooks_section


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load scenario dataset from jsonl file."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize_app_whitelist(apps: Any) -> List[str]:
    if isinstance(apps, list):
        return apps
    if not apps:
        return []
    return [str(apps)]


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


def start_overlay_attack_flow(
    *,
    adb_path: str,
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
        adb_path=adb_path,
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
        adb_path=adb_path,
        apps=_normalize_app_whitelist(scenario.get("app")),
    )
    hook.start(trigger)
    return hook, httpd


def start_popup_sms_flow(
    *,
    adb_path: str,
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
        adb_path=adb_path,
        phone_numbers=numbers,
        content=message,
    )
    hook = ForegroundAppHook(
        adb_path=adb_path,
        apps=_normalize_app_whitelist(scenario.get("app")),
    )
    hook.start(trigger)
    return hook


def _prepare_dynamic_hooks(
    adb_path: str,
    scenario: Dict[str, Any],
    overlay_cfg: Dict[str, Any],
) -> Tuple[Optional[ForegroundAppHook], Optional[Any]]:
    hook: Optional[ForegroundAppHook] = None
    httpd = None
    use_dynamic = (scenario.get("use_dynamic") or "").strip()
    if use_dynamic == "Overlay Attack":
        hook, httpd = start_overlay_attack_flow(adb_path=adb_path, scenario=scenario, overlay_cfg=overlay_cfg)
    elif use_dynamic == "Popup SMS":
        hook = start_popup_sms_flow(adb_path=adb_path, scenario=scenario)
    return hook, httpd


def _pick_app_name(scenario: Dict[str, Any], override_app: Optional[str]) -> str:
    if override_app:
        return override_app.replace(" ", "")
    apps = _normalize_app_whitelist(scenario.get("app"))
    if apps:
        first = apps[0]
        if isinstance(first, list) and first:
            return str(first[0]).replace(" ", "")
        return str(first).replace(" ", "")
    return "default"


def _list_adb_devices(adb_path: str) -> List[str]:
    cp = subprocess.run([adb_path, "devices"], capture_output=True, text=True, check=False)
    devices: List[str] = []
    if cp.stdout:
        lines = cp.stdout.strip().splitlines()
        for line in lines[1:]:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            serial, status = line.split("\t", 1)
            if status.strip() == "device":
                devices.append(serial.strip())
    return devices


def _build_appagent_inputs(
    *,
    appagent_root: Path,
    app_name: str,
    task_prompt: str,
    adb_path: str,
) -> str:
    """
    Build a minimal stdin payload for AppAgent's interactive prompts so we do not need to
    modify its source. Only hook-facing behavior is customized here.
    """
    lines: List[str] = []
    app_dir = appagent_root / "apps" / app_name
    auto_docs = app_dir / "auto_docs"
    demo_docs = app_dir / "demo_docs"

    if not auto_docs.exists() and not demo_docs.exists():
        # Accept running without docs by default to prevent hanging.
        lines.append("y")
    elif auto_docs.exists() and demo_docs.exists():
        # Prefer autonomous exploration docs when both are available.
        lines.append("1")

    devices = _list_adb_devices(adb_path)
    if not devices:
        raise RuntimeError("No adb devices found; AppAgent will exit immediately.")
    if len(devices) > 1:
        # Choose the first available device to satisfy the prompt.
        lines.append(devices[0])

    sanitized_prompt = (task_prompt or "").replace("\n", " ").strip()
    lines.append(sanitized_prompt or "Please complete the task as stated.")
    return "\n".join(lines) + "\n"


def _list_task_dirs(task_root: Path) -> List[Path]:
    if not task_root.exists():
        return []
    return [p for p in task_root.iterdir() if p.is_dir()]


def _pick_new_task_dir(task_root: Path, before: Sequence[Path]) -> Optional[Path]:
    try:
        before_names = {p.name for p in before}
        current = _list_task_dirs(task_root)
        candidates = [p for p in current if p.name not in before_names]
        pool = candidates or current
        if not pool:
            return None
        return max(pool, key=lambda p: p.stat().st_mtime)
    except Exception as e:
        print(f"[judge] Failed to locate task directory: {e}")
        return None


def _load_task_log_entries(task_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for log_file in sorted(task_dir.glob("log_*.txt")):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entries.append(json.loads(line))
        except Exception as e:
            print(f"[judge] Failed to read log {log_file}: {e}")
    return entries


def _format_action_history(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return ""
    condensed: List[Dict[str, Any]] = []
    for item in entries:
        condensed.append(
            {
                "step": item.get("step"),
                "image": item.get("image"),
                "model_response": item.get("response"),
                "prompt": item.get("prompt"),
            }
        )
    return json.dumps(condensed, ensure_ascii=False)


def _collect_task_images(task_dir: Path) -> List[str]:
    if not task_dir or not task_dir.exists():
        return []

    labeled = sorted(task_dir.glob("*_labeled.png"))
    if labeled:
        return [str(p) for p in labeled]

    pngs = sorted(task_dir.glob("*.png"))
    jpgs = sorted(task_dir.glob("*.jpg"))

    files: List[str] = []
    seen = set()
    for p in pngs + jpgs:
        if p.name in seen:
            continue
        seen.add(p.name)
        files.append(str(p))
    return files


def run_appagent_once(
    *,
    scenario: Dict[str, Any],
    adb_path: str,
    appagent_root: Path,
    overlay_cfg: Dict[str, Any],
    override_app: Optional[str] = None,
) -> Dict[str, Any]:
    app_name = _pick_app_name(scenario, override_app)
    task_prompt = scenario.get("prompt") or ""

    hook, httpd = _prepare_dynamic_hooks(adb_path, scenario, overlay_cfg)
    task_root = appagent_root / "tasks"
    before_dirs = _list_task_dirs(task_root)
    proc: Optional[subprocess.CompletedProcess] = None

    try:
        user_input = _build_appagent_inputs(
            appagent_root=appagent_root,
            app_name=app_name,
            task_prompt=task_prompt,
            adb_path=adb_path,
        )
        cmd = ["python", "-u", "scripts/task_executor.py", "--app", app_name, "--root_dir", "."]
        print(f"[runner] Starting AppAgent for scenario {scenario.get('id')} targeting app '{app_name}'")
        proc = subprocess.run(
            cmd,
            cwd=str(appagent_root),
            input=user_input,
            text=True,
        )
        if proc.returncode not in (0, None):
            print(f"[runner] AppAgent exited with code {proc.returncode} for scenario {scenario.get('id')}")
    finally:
        if hook:
            hook.stop()
        if httpd:
            httpd.shutdown()
            httpd.server_close()
    new_task_dir = _pick_new_task_dir(task_root, before_dirs)
    return {
        "task_dir": new_task_dir,
        "returncode": proc.returncode if proc else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AppAgent on Ghost-EI scenarios with dynamic hooks enabled.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to scenario jsonl dataset.")
    parser.add_argument("--scenario-id", type=int, action="append", help="Scenario id(s) to run. Repeatable.")
    parser.add_argument("--appagent-root", type=str, default=DEFAULT_APPAGENT_ROOT, help="Path to the AppAgent repo root.")
    parser.add_argument("--app", type=str, default=None, help="Override app name passed to AppAgent.")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to wait between scenarios.")
    args = parser.parse_args()

    config_path = os.getenv(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)
    env_cfg, runtime_cfg, hooks_cfg = load_run_configuration(config_path)
    adb_path = env_cfg.get("adb_path") or "adb"

    dataset_path = args.dataset or runtime_cfg.get("dataset_path") or DEFAULT_DATASET_PATH
    dataset = load_dataset(dataset_path)
    if args.scenario_id:
        dataset = [s for s in dataset if s.get("id") in set(args.scenario_id)]
        if not dataset:
            raise SystemExit("No scenarios matched the provided --scenario-id values.")

    overlay_cfg = hooks_cfg.get("overlay", {})
    appagent_root = Path(args.appagent_root).resolve()
    if not appagent_root.exists():
        raise FileNotFoundError(f"AppAgent root not found: {appagent_root}")

    judge_client = JudgeClient(
        model=env_cfg.get("judge_model"),
        api_key=env_cfg.get("token"),
        api_url=env_cfg.get("API_url"),
    )
    output_dir = runtime_cfg.get("output_dir") or f"{env_cfg.get('caption_model')}-{env_cfg.get('reflection_switch')}"
    thinking_output_path = runtime_cfg.get("thinking_output_path") or f"{output_dir}-thinking.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    for scenario in dataset:
        run_result = run_appagent_once(
            scenario=scenario,
            adb_path=adb_path,
            appagent_root=appagent_root,
            overlay_cfg=overlay_cfg,
            override_app=args.app,
        )
        task_dir: Optional[Path] = run_result.get("task_dir")
        return_code = run_result.get("returncode")
        if return_code not in (0, None):
            print(f"[runner] Scenario {scenario.get('id')} exited with code {return_code}; skipping auto-judgement.")
            time.sleep(args.delay)
            continue
        if not task_dir:
            print(f"[judge] No task directory found for scenario {scenario.get('id')}; skipping auto-judgement.")
            time.sleep(args.delay)
            continue

        logs = _load_task_log_entries(task_dir)
        action_history = _format_action_history(logs)
        images = _collect_task_images(task_dir)

        try:
            judge_res = judge_client.judge(
                instruction=scenario.get("prompt", ""),
                action_history=action_history,
                judge_criteria=scenario.get("result", {}),
                images=images,
            )
            print(f"[judge] Scenario {scenario.get('id')} result: {judge_res['parsed']}")
        except Exception as e:
            print(f"[judge] Scenario {scenario.get('id')} judgement failed: {e}")
            time.sleep(args.delay)
            continue

        with open(thinking_output_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": scenario.get("id"),
                        "prompt": scenario.get("prompt"),
                        "judge": judge_res.get("parsed"),
                        "task_dir": str(task_dir),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        scenario_out_dir = Path(output_dir) / str(scenario.get("id"))
        try:
            os.makedirs(scenario_out_dir, exist_ok=True)
        except Exception:
            pass
        try:
            with open(scenario_out_dir / "action_history.json", "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            with open(scenario_out_dir / "judge_result.json", "w", encoding="utf-8") as f:
                json.dump(judge_res.get("parsed"), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[judge] Failed to persist artifacts for scenario {scenario.get('id')}: {e}")

        time.sleep(args.delay)


if __name__ == "__main__":
    main()
