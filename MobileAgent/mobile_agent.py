# -*- coding: utf-8 -*-

import subprocess
import os
import time
import copy
import torch
import shutil
import concurrent.futures
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import base64
import openai
import stat
from typing import List, Dict, Any, Optional, Sequence, Union
from PIL import Image, ImageDraw
import openai
# === External dependencies (kept consistent with original script) ===
from MobileAgent.api import inference_chat, inference_reasoning_chat
from MobileAgent.text_localization import ocr
from MobileAgent.icon_localization import det
from MobileAgent.controller import get_screenshot, tap, slide, type, back, home, take_screenshot
from MobileAgent.android_env import AndroidEnv, EnvConfig
from MobileAgent.prompt import (
    get_action_prompt,
    get_reflect_prompt,
    get_memory_prompt,
    get_process_prompt,
)
from MobileAgent.chat import (
    init_action_chat,
    init_reflect_chat,
    init_memory_chat,
    add_response,
    add_response_two_image,
)

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dashscope import MultiModalConversation
import dashscope

# =========================
# Agent class: orchestrates loop, conversations, execution, reflection/memory/planning
# =========================
class MobileAgent:
    def __init__(self, env_inst: AndroidEnv, instruction: str):
        self.e = env_inst
        self.instruction = instruction

        # Runtime state
        self.output_action_history: List[str] = []
        self.thought_history: List[str] = []
        self.summary_history: List[str] = []
        self.action_history: List[str] = []
        self.summary: str = ""
        self.action: str = ""
        self.completed_requirements: str = ""
        self.memory: str = ""
        self.insight: str = ""
        self.error_flag: bool = False

        # Perception cache (prevents UnboundLocalError)
        self.perception_infos: List[Dict[str, Any]] | None = None
        self.width: int | None = None
        self.height: int | None = None

        # Keyboard state
        self.keyboard: bool = False
        self.keyboard_height_limit: int = 0

        # Paths
        self.screenshot_file = os.path.join(self.e.cfg.screenshot_dir, "screenshot.jpg")
        self.last_screenshot_file = os.path.join(self.e.cfg.screenshot_dir, "last_screenshot.jpg")

    # ---------- Parsing helpers ----------
    @staticmethod
    def _safe_extract(text: str, start: str, end: str = None, default: str = "") -> str:
        try:
            if end:
                seg = text.split(start, 1)[-1].split(end, 1)[0]
            else:
                seg = text.split(start, 1)[-1]
            return seg.replace("\n", " ").replace("  ", " ").strip()
        except Exception:
            return default

    def _parse_action_output(self, output_action: str) -> Tuple[str, str, str]:
        thought = self._safe_extract(output_action, "### Thought ###", "### Action ###")
        action = self._safe_extract(output_action, "### Action ###", "### Operation ###")
        summary = self._safe_extract(output_action, "### Operation ###")
        return thought, action, summary

    # ---------- Action execution ----------
    def _execute_action(self, action: str) -> bool:
        """
        Execute an action string parsed from the language model.
        """
        if "Open app" in action:
            app_name = action.split("(")[-1].split(")")[0]
            texts, boxes = ocr(self.screenshot_file, self.e.ocr_detection, self.e.ocr_recognition)
            for ti in range(len(texts)):
                if app_name == texts[ti]:
                    # Tap the app icon slightly above the text box
                    cx = int((boxes[ti][0] + boxes[ti][2]) / 2)
                    cy = int((boxes[ti][1] + boxes[ti][3]) / 2)
                    tap(self.e.cfg.adb_path, cx, cy - int(boxes[ti][3] - boxes[ti][1]))
                    break

        elif "Tap" in action:
            coord = action.split("(")[-1].split(")")[0].split(",")
            x, y = int(coord[0]), int(coord[1])
            tap(self.e.cfg.adb_path, x, y)

        elif "Swipe" in action:
            c1 = action.split("Swipe (")[-1].split("), (")[0].split(",")
            c2 = action.split("), (")[-1].split(")")[0].split(",")
            x1, y1 = int(c1[0]), int(c1[1])
            x2, y2 = int(c2[0]), int(c2[1])
            slide(self.e.cfg.adb_path, x1, y1, x2, y2)

        elif 'Type' in action:
            text = ""
            if '<text>' in action:
                # Extract text inside the <text> tags

                text = action.split('<text>')[-1].split('</text>')[0]
            else:
                text = action.split('Type')[-1]

            type(self.e.cfg.adb_path, text.strip())

        elif 'Back' in action:
            back(self.e.cfg.adb_path)

        elif 'Home' in action:
            home(self.e.cfg.adb_path)
        elif 'Enter' in action:
            type(self.e.cfg.adb_path, '\n')
        elif 'ScreenShot' in action:
            take_screenshot(self.e.cfg.adb_path)
            return True
        elif 'Stop' in action:
            return False  # Stop execution
        
        return True  # Continue execution
        
    
    # ---------- Ensure current perception ----------
    def _ensure_current_perception(self):
        """
        Guarantee self.perception_infos / self.width / self.height are populated.
        """
        if self.perception_infos is None or self.width is None or self.height is None:
            self.perception_infos, self.width, self.height = self.e.get_perception_infos(self.screenshot_file)
            self.keyboard_height_limit = int(0.9 * self.height)
            self.keyboard = self.e.keyboard_present(self.perception_infos, self.height, 0.9)

    # ---------- Single iteration ----------
    def _step(self, iter_idx: int) -> bool:
        # Ensure the latest perception snapshot is available before the step
        self._ensure_current_perception()

        # === Action phase ===
        prompt_action = get_action_prompt(
            self.instruction,
            self.perception_infos,
            self.width,
            self.height,
            self.keyboard,
            self.summary_history,
            self.action_history,
            self.summary,
            self.action,
            self.e.cfg.add_info,
            self.error_flag,
            self.completed_requirements,
            self.memory,
        )
        chat_action = init_action_chat()
        chat_action = add_response("user", prompt_action, chat_action, self.screenshot_file)
        if self.e.cfg.reasoning_switch:
            output_action = inference_reasoning_chat(chat_action, self.e.cfg.caption_model, self.e.cfg.API_url, self.e.cfg.token)
        else:
            output_action = inference_chat(chat_action, self.e.cfg.caption_model, self.e.cfg.API_url, self.e.cfg.token)
        thought, action, summary = self._parse_action_output(output_action)
        chat_action = add_response("assistant", output_action, chat_action)
        self.insight = "\n".join([
            '### Instruction ###\n' + self.instruction,
            '### Thought ###\n' + thought,
            '### Action ###\n' + action,
            '### Summary ###\n' + summary,
            '### Perception infos ###\n' + str(self.perception_infos),
        ])

        print("#" * 50 + " Decision " + "#" * 50)
        print(output_action)
        print("#" * 114)

        self.summary, self.action = summary, action

        # === Memory phase (optional) ===
        if self.e.cfg.memory_switch:
            prompt_memory = get_memory_prompt(self.insight)
            chat_action = add_response("user", prompt_memory, chat_action)
            output_memory = inference_chat(chat_action, self.e.cfg.caption_model, self.e.cfg.API_url, self.e.cfg.token)
            chat_action = add_response("assistant", output_memory, chat_action)

            print("#" * 50 + " Memory " + "#" * 50)
            print(output_memory)
            print("#" * 114)

            important = self._safe_extract(output_memory, "### Important content ###", "\n\n")
            if important and "None" not in important and (important + "\n") not in self.memory:
                self.memory += important + "\n"

        # === Execute action ===
        keep_running = self._execute_action(action)
        if not keep_running:
            return False
        if "Sorry" in output_action or "sorry" in output_action:
            return False
        time.sleep(5)

        # === Persist previous frame and refresh current frame ===
        last_perception_infos = copy.deepcopy(self.perception_infos)

        # Rotate screenshot files (current screenshot.jpg -> last_screenshot.jpg)
        if os.path.exists(self.last_screenshot_file):
            os.remove(self.last_screenshot_file)
        os.replace(self.screenshot_file, self.last_screenshot_file)

        # Fetch fresh perception data as the new current frame
        self.perception_infos, self.width, self.height = self.e.get_perception_infos(self.screenshot_file)
        self.keyboard = self.e.keyboard_present(self.perception_infos, self.height, 0.9)

        # === Reflection phase (optional) ===
        if self.e.cfg.reflection_switch:
            prompt_reflect = get_reflect_prompt(
                self.instruction,
                last_perception_infos,
                self.perception_infos,
                self.width,
                self.height,
                self.e.keyboard_present(last_perception_infos, self.height, 0.9),
                self.keyboard,
                self.summary,
                self.action,
                self.e.cfg.add_info,
            )
            chat_reflect = init_reflect_chat()
            chat_reflect = add_response_two_image(
                "user", prompt_reflect, chat_reflect, [self.last_screenshot_file, self.screenshot_file]
            )

            output_reflect = inference_chat(chat_reflect, self.e.cfg.reflect_model, self.e.cfg.API_url, self.e.cfg.token)
            reflect = self._safe_extract(output_reflect, "### Answer ###")
            chat_reflect = add_response("assistant", output_reflect, chat_reflect)

            print("#" * 50 + " Reflcetion " + "#" * 50)
            print(output_reflect)
            print("#" * 114)

            if "A" in reflect:
                # Action succeeded; update process memory and plan progress
                self.thought_history.append(thought)
                self.summary_history.append(summary)
                self.action_history.append(action)
                self.output_action_history.append(output_action)

                prompt_plan = get_process_prompt(
                    self.instruction,
                    self.thought_history,
                    self.summary_history,
                    self.action_history,
                    self.completed_requirements,
                    self.e.cfg.add_info,
                )
                chat_plan = init_memory_chat()
                chat_plan = add_response("user", prompt_plan, chat_plan)
                output_plan = inference_chat(chat_plan, self.e.cfg.caption_model, self.e.cfg.API_url, self.e.cfg.token)
                chat_plan = add_response("assistant", output_plan, chat_plan)

                print("#" * 50 + " Planning " + "#" * 50)
                print(output_plan)
                print("#" * 114)

                self.completed_requirements = self._safe_extract(output_plan, "### Completed contents ###")
                self.error_flag = False

            elif "B" in reflect:
                self.error_flag = True
                back(self.e.cfg.adb_path)

            elif "C" in reflect:
                self.error_flag = True

        else:
            # Skip reflection and plan directly
            self.thought_history.append(thought)
            self.summary_history.append(summary)
            self.action_history.append(action)
            self.output_action_history.append(output_action)

            prompt_plan = get_process_prompt(
                self.instruction,
                self.thought_history,
                self.summary_history,
                self.action_history,
                self.completed_requirements,
                self.e.cfg.add_info,
            )
            chat_plan = init_memory_chat()
            chat_plan = add_response("user", prompt_plan, chat_plan)
            output_plan = inference_chat(chat_plan, self.e.cfg.caption_model, self.e.cfg.API_url, self.e.cfg.token)
            chat_plan = add_response("assistant", output_plan, chat_plan)

            print("#" * 50 + " Planning " + "#" * 50)
            print(output_plan)
            print("#" * 114)

            self.completed_requirements = self._safe_extract(output_plan, "### Completed contents ###")

        # Remove the temporary last screenshot
        if os.path.exists(self.last_screenshot_file):
            os.remove(self.last_screenshot_file)

        return True
    def clear(self, base_dir: Optional[str] = None) -> Dict[str, Any]:
            """
            Delete ALL contents under the 'judgement' directory (but keep the directory itself).
            By default, 'judgement' is resolved relative to:
            - base_dir if provided, else the current working directory.

            Returns:
                {
                "path": "<absolute path to judgement>",
                "deleted": <int number of entries removed>,
                "errors": [ "path: error message", ... ],
                "exists": <bool whether judgement existed>
                }
            """
            root = base_dir or os.getcwd()
            target = os.path.abspath(root)

            result = {"path": target, "deleted": 0, "errors": [], "exists": os.path.exists(target)}

            # Safety guard: only operate if it's exactly a directory named 'judgement'
            if os.path.basename(target.rstrip(os.sep)) != "judgement":
                result["errors"].append(f"Refusing to clear a non-'judgement' path: {target}")
                return result

            if not os.path.isdir(target):
                # nothing to clear (either doesn't exist or not a dir)
                return result

            def _on_rm_error(func, path, exc_info):
                # Handle read-only files on Windows/Linux
                try:
                    os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                    func(path)
                except Exception as e:
                    result["errors"].append(f"{path}: {e}")

            # Iterate and remove all items inside 'judgement'
            with os.scandir(target) as it:
                for entry in it:
                    try:
                        p = entry.path
                        if entry.is_dir(follow_symlinks=False):
                            shutil.rmtree(p, onerror=_on_rm_error)
                        else:
                            # file or symlink
                            try:
                                os.chmod(p, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                            except Exception:
                                pass
                            os.remove(p)
                        result["deleted"] += 1
                    except Exception as e:
                        result["errors"].append(f"{entry.path}: {e}")

            return result
    # ---------- Main loop ----------
    def run(self, max_iters: int = 25):
        try:
            for i in range(1, max_iters + 1):
                try:
                    cont = self._step(i)
                    if not cont:
                        break
                except:
                    time.sleep(2)
        except KeyboardInterrupt:
            self.clear("judgement")
