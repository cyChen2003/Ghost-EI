from __future__ import annotations

import base64
import mimetypes
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, cast
import time
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from .actions import (
    MobileAction,
    make_open_app_action,
    make_swipe_action,
    make_tap_action,
    make_type_action,
    normalize_point,
    simple_action,
)


def _encode_file_to_data_url(path: str) -> str:
    """Encode a file to a data URL for embedding in HTTP requests."""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    with open(path, "rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _flatten_choice_content(content: Any) -> str:
    """Flatten choice content from API response to a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for chunk in content:
            if isinstance(chunk, dict):
                text = chunk.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(chunk, str):
                parts.append(chunk)
        return "".join(parts).strip()
    return str(content)


class _RemoteVLLMClient:
    """Thin HTTP client for OpenAI-compatible vLLM endpoints."""

    def __init__(
        self,
        *,
        api_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        timeout: float = 120.0,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.api_url = self._normalize_api_url(api_url)
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.extra_params = extra_params or {}
        self.timeout = timeout
        self.user = user
        self.password = password
        self._auth_header = None
        self.content = None
        if user and password:
            token = f"{user}:{password}".encode("utf-8")
            b64 = base64.b64encode(token).decode("utf-8")
            print(f"the b64 is {b64}")  # Debug output, can be ignored
            self._auth_header = {"Authorization": f"Basic {b64}"}

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert messages to OpenAI-compatible format, encoding image paths as data URLs."""
        converted: List[Dict[str, Any]] = []
        for message in messages:
            chunks = []
            for chunk in message.get("content", []):
                chunk_type = chunk.get("type")
                if chunk_type == "text":
                    chunks.append({"type": "text", "text": chunk.get("text", "")})
                elif chunk_type in {"image", "image_path"}:
                    image_path = chunk.get("image") or chunk.get("image_path")
                    if not image_path:
                        raise ValueError("Image chunk requires an 'image' or 'image_path' field.")
                    chunks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": _encode_file_to_data_url(image_path)},
                        }
                    )
                else:
                    chunks.append(chunk)
            converted.append({"role": message.get("role", "user"), "content": chunks})
        return converted

    def generate(self, messages: List[Dict[str, Any]], max_tokens: int) -> str:
        """Generate completion from remote vLLM endpoint using HTTP Basic auth if configured."""
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._convert_messages(messages),
            "max_tokens": max_tokens,
        }
        
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.extra_params:
            payload.update(self.extra_params)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self._auth_header:
            headers.update(self._auth_header)
        try:
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
        except requests.exceptions.SSLError as exc:
            raise RuntimeError(
                f"SSL handshake failed when calling remote endpoint {self.api_url}: {exc}. "
                "Ensure the URL is reachable from this machine and that its certificate chain is trusted.",
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"HTTP error when calling remote endpoint {self.api_url}: {exc}") from exc
        response.raise_for_status()
        data = response.json()
        print(data)
        try:
            content = data["choices"][0]["message"]["content"]
            self.content = content
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected response schema from remote vLLM: {data}") from exc
        return _flatten_choice_content(content)

    @staticmethod
    def _normalize_api_url(api_url: str) -> str:
        """Normalize API URL to ensure it ends with /v1/chat/completions or similar endpoint."""
        endpoint = str(api_url or "").strip()
        if not endpoint:
            raise ValueError("remote_api_url must be provided for remote inference.")
        endpoint = endpoint.rstrip("/")
        lowered = endpoint.lower()
        suffixes = ("/chat/completions", "/completions", "/responses")
        if any(lowered.endswith(sfx) for sfx in suffixes):
            return endpoint
        if lowered.endswith("/v1"):
            return f"{endpoint}/chat/completions"
        return f"{endpoint}/v1/chat/completions"


def _coerce_dtype(dtype: Any) -> Any:
    """Convert string dtype to torch dtype if needed."""
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


class BaseGUIAgent:
    """Common utilities shared across all fine-tuned GUI agents."""

    def __init__(
        self,
        model_cls,
        model_path: str,
        processor_path: Optional[str] = None,
        processor_cls=AutoProcessor,
        device: str = "cuda",
        torch_dtype: Any = "bfloat16",
        max_new_tokens: int = 128,
        model_kwargs: Optional[Dict[str, Any]] = None,
        processor_kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        inference_mode: Literal["local", "remote"] = "local",
        remote_api_url: Optional[str] = None,
        remote_model: Optional[str] = None,
        remote_api_key: Optional[str] = None,
        remote_temperature: Optional[float] = None,
        remote_top_p: Optional[float] = None,
        remote_extra_params: Optional[Dict[str, Any]] = None,
        remote_timeout: float = 120.0,

    ) -> None:
        processor_path = processor_path or model_path
        model_kwargs = model_kwargs or {"device_map": "auto", "attn_implementation": "flash_attention_2"}
        processor_kwargs = processor_kwargs or {"use_fast": True}

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.name = name or self.__class__.__name__
        mode = (inference_mode or "local").lower()
        if mode not in {"local", "remote"}:
            raise ValueError(f"Unsupported inference mode: {inference_mode}")

        self._inference_mode = cast(Literal["local", "remote"], mode)
        self.model = None
        self.processor = None
        self._remote_client: Optional[_RemoteVLLMClient] = None
        self.raw_action = None
        self.output_text = None
        if mode == "local":
            self.model = model_cls.from_pretrained(
                model_path,
                torch_dtype=_coerce_dtype(torch_dtype),
                **model_kwargs,
            )
            self.processor = processor_cls.from_pretrained(processor_path, **processor_kwargs)
        else:
            if not remote_api_url or not remote_model:
                raise ValueError("Remote mode requires both --remote-api-url and --remote-model parameters.")
            self._remote_client = _RemoteVLLMClient(
                api_url=remote_api_url,
                api_key=remote_api_key,
                model_name=remote_model,
                temperature=remote_temperature,
                top_p=remote_top_p,
                extra_params=remote_extra_params,
                timeout=remote_timeout,

            )

    def build_messages(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build messages for model inference from observation dictionary. Must be implemented by subclasses."""
        raise NotImplementedError

    def extract_action_from_output(self, model_output: str) -> str:
        """Extract action string from model output. Must be implemented by subclasses."""
        raise NotImplementedError

    def map_raw_action(self, raw_action: str, obs: Dict[str, Any]) -> MobileAction:
        """Convert raw action string to MobileAction. Must be implemented by subclasses."""
        raise NotImplementedError

    def _generate_raw_action(self, obs: Dict[str, Any]) -> str:
        """Generate raw action string from observation using local or remote inference."""
        messages = self.build_messages(obs)
        if self._inference_mode == "remote":
            if not self._remote_client:
                raise RuntimeError("Remote inference requested but no client is configured.")
            output_text = self._remote_client.generate(messages, max_tokens=self.max_new_tokens)
        else:
            output_text = self._generate_local_completion(messages)
        self.output_text = output_text
        return self.extract_action_from_output(output_text)

    def _generate_local_completion(self, messages: List[Dict[str, Any]]) -> str:
        """Generate completion using local HuggingFace model."""
        if not self.processor or not self.model:
            raise RuntimeError("Local inference requires both model and processor to be initialized.")
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        return output_text

    def predict_action(self, obs: Dict[str, Any], return_mobile_action: bool = False) -> str | MobileAction:
        """Predict action from observation, returning either command string or MobileAction object."""
        raw_action = self._generate_raw_action(obs)
        self.raw_action = raw_action
        mobile_action = self.map_raw_action(raw_action, obs)
        if return_mobile_action:
            return mobile_action
        return mobile_action.to_command()


ATLAS_PROMPT = """
You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

1. Basic Actions
Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
Basic Action 1: CLICK 
    - purpose: Click at the specified position.
    - format: CLICK <point>[[x-axis, y-axis]]</point>
    - example usage: CLICK <point>[[101, 872]]</point>
       
Basic Action 2: TYPE
    - purpose: Enter specified text at the designated location.
    - format: TYPE [input text]
    - example usage: TYPE [Shanghai shopping mall]

Basic Action 3: SCROLL
    - purpose: SCROLL in the specified direction.
    - format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
    - example usage: SCROLL [UP]
    
2. Custom Actions
Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.
Custom Action 1: LONG_PRESS 
    - purpose: Long press at the specified position.
    - format: LONG_PRESS <point>[[x-axis, y-axis]]</point>
    - example usage: LONG_PRESS <point>[[101, 872]]</point>
       
Custom Action 2: OPEN_APP
    - purpose: Open the specified application.
    - format: OPEN_APP [app_name]
    - example usage: OPEN_APP [Google Chrome]

Custom Action 3: PRESS_BACK
    - purpose: Press a back button to navigate to the previous screen.
    - format: PRESS_BACK
    - example usage: PRESS_BACK

Custom Action 4: PRESS_HOME
    - purpose: Press a home button to navigate to the home page.
    - format: PRESS_HOME
    - example usage: PRESS_HOME

Custom Action 5: WAIT
    - purpose: Wait for the screen to load.
    - format: WAIT
    - example usage: WAIT

Custom Action 6: COMPLETE
    - purpose: Indicate the task is finished.
    - format: COMPLETE
    - example usage: COMPLETE

In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action. Ensure you strictly generate two sections: Thoughts and Actions.
Thoughts: Clearly outline your reasoning process for current step.
Actions: Specify the actual actions you will take based on your reasoning. You should follow action format above when generating. 

If you see "Actions already completed" below, those steps have been executed; the screenshot shows the state AFTER them. Do NOT repeat the same action. Choose the NEXT different action or COMPLETE if the task is done.

Your current task instruction, action history, and associated screenshot are as follows:

""".strip()


class OSAtlasAgent(BaseGUIAgent):
    """OS-Atlas agent using Qwen2VL model for GUI task execution."""
    
    def __init__(self, model_path: str, processor_path: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(
            model_cls=Qwen2VLForConditionalGeneration,
            model_path=model_path,
            processor_path=processor_path,
            **kwargs,
        )

    def build_messages(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        instruction = obs.get("task") or obs.get("query_rewritten") or ""
        step_info = obs.get("step_list_rewritten")
        if step_info:
            extra_text = (
                "### Previous Actions\n"
                "Your previous outputs (thoughts and actions) — the screenshot is the state AFTER these; do not repeat them. Decide only the NEXT action.\n"
                "Previous steps:\n"
                f"{step_info}\n\n"
            )
        else:
            extra_text = ""
        content = f"\nTask instruction: {instruction}\n{extra_text}Screenshot: \n"
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ATLAS_PROMPT + content},
                    {"type": "image", "image": obs["image_path"]},
                ],
            }
        ]

    def extract_action_from_output(self, model_output: str) -> str:
        match = re.search(r"actions:\s*(.*?)(?:<\|im_end\|>|$)", model_output, re.IGNORECASE | re.DOTALL)
        if not match:
            # Fallback to singular "Action:" schema used by some compatible endpoints.
            match = re.search(r"action:\s*(.*?)(?:<\|im_end\|>|$)", model_output, re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError("No action block found in model output.")
        time.sleep(0.2)
        action_block = match.group(1).strip()
        for line in action_block.splitlines():
            cleaned = line.strip()
            if cleaned:
                return cleaned
        raise ValueError("Action block is empty in model output.")


class BaseTarsAgent(BaseGUIAgent):
    """Base class for TARS agents with common message building and action extraction logic."""
    
    MODEL_CLASS = Qwen2VLForConditionalGeneration

    def __init__(self, model_path: str, processor_path: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(
            model_cls=self.MODEL_CLASS,
            model_path=model_path,
            processor_path=processor_path,
            **kwargs,
        )

    def build_messages(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = build_tars_prompt(obs)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": obs["image_path"]},
                ],
            }
        ]

    def extract_action_from_output(self, model_output: str) -> str:
        pattern = r"Action:\s*(.*?)(?:<\|im_end\|>|\n|$)"
        match = re.search(pattern, model_output, re.IGNORECASE)
        if not match:
            raise ValueError("No Action clause found in model output.")
        return match.group(1).strip()

    def map_raw_action(self, raw_action: str, obs: Dict[str, Any]) -> MobileAction:
        return tars_to_mobile(raw_action, obs.get("image_path"))


class UITarsAgent(BaseTarsAgent):
    """UI-TARS 7B SFT model."""

    MODEL_CLASS = Qwen2VLForConditionalGeneration


class TARS15Agent(BaseTarsAgent):
    """TARS 1.5 (Qwen2.5 VL) model."""

    MODEL_CLASS = Qwen2_5_VLForConditionalGeneration

def atlas_to_mobile(raw_action: str) -> MobileAction:
    """Convert ATLAS action format to MobileAction."""
    action_line = _first_action_line(raw_action)
    upper_line = action_line.upper()

    if upper_line == "PRESS_BACK":
        return simple_action("Back")
    if upper_line == "PRESS_HOME":
        return simple_action("Home")
    if upper_line == "WAIT":
        return simple_action("Wait")
    if upper_line == "COMPLETE":
        return simple_action("Stop")

    if upper_line.startswith("SCROLL"):
        direction_match = re.search(r"SCROLL\s*\[(UP|DOWN|LEFT|RIGHT)\]", upper_line)
        direction = direction_match.group(1) if direction_match else "DOWN"
        return make_swipe_action(direction)
    if upper_line.startswith("OPEN_APP"):
        app_match = re.search(r"OPEN_APP\s*\[(.*?)\]", action_line, re.IGNORECASE | re.DOTALL)
        app_name = app_match.group(1).strip() if app_match else action_line.split("OPEN_APP", 1)[-1].strip("[] ")
        return make_open_app_action(app_name)
    coord = _extract_bracket_coordinates(action_line)
    if coord and (upper_line.startswith("CLICK") or upper_line.startswith("LONG_PRESS")):
        return make_tap_action(*coord)

    if upper_line.startswith("TYPE"):
        text_match = re.search(r"TYPE\s*\[(.*)\]", action_line, re.IGNORECASE)
        text = text_match.group(1) if text_match else action_line.split("TYPE", 1)[-1].strip()
        return make_type_action(text.strip("[] "))

    # Fallback: return Stop action if no pattern matches
    return simple_action("Stop")


def tars_to_mobile(raw_action: str, image_path: Optional[str]) -> MobileAction:
    """Convert TARS action format to MobileAction."""
    action_line = _first_action_line(raw_action)
    upper_line = action_line.upper()

    if upper_line == "PRESS_HOME()":
        return simple_action("Home")
    if upper_line == "PRESS_BACK()":
        return simple_action("Back")
    if upper_line.startswith("FINISHED"):
        return simple_action("Stop")
    if upper_line.startswith("WAIT"):
        return simple_action("Wait")
    if upper_line.startswith("SCROLL"):
        direction_match = re.search(r"direction=['\"](UP|DOWN|LEFT|RIGHT)['\"]", upper_line, re.IGNORECASE)
        direction = direction_match.group(1) if direction_match else "DOWN"
        return make_swipe_action(direction)

    if upper_line.startswith("TYPE"):
        text_match = re.search(r"content='([^']*)'", action_line, re.IGNORECASE)
        text = text_match.group(1) if text_match else ""
        return make_type_action(text)

    coord = _extract_tars_coordinates(action_line, image_path)
    if coord and (upper_line.startswith("CLICK") or upper_line.startswith("LONG_PRESS")):
        return make_tap_action(*coord)

    return simple_action("Wait")


def _first_action_line(raw_action: str) -> str:
    """Extract the first non-empty line from raw action string."""
    for line in raw_action.strip().splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return raw_action.strip()


def _extract_bracket_coordinates(action_line: str) -> Optional[Tuple[int, int]]:
    """Extract coordinates from ATLAS action format: [[x, y]]."""
    match = re.search(r"\[\[(\d+),\s*(\d+)\]\]", action_line)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _extract_tars_coordinates(action_line: str, image_path: Optional[str]) -> Optional[Tuple[int, int]]:
    """Extract coordinates from TARS action string, preferring normalized coordinates aligned with MobileAgent (0-1000)."""
    candidates = [
        re.search(r"point=['\"]<point>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</point>['\"]", action_line, re.IGNORECASE),
        re.search(r"point=['\"](\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)['\"]", action_line, re.IGNORECASE),
        re.search(r"start_box=['\"]?\(?(\d+(?:\.\d+)?)[,\s]+(\d+(?:\.\d+)?)\)?['\"]?", action_line, re.IGNORECASE),
        re.search(r"\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)", action_line),
    ]
    for match in candidates:
        if match:
            try:
                raw_x, raw_y = float(match.group(1)), float(match.group(2))
            except ValueError:
                continue
            if image_path:
                width, height = _safe_image_size(image_path)
            else:
                width = height = None
            norm_x, norm_y = normalize_point(raw_x, raw_y, width, height, normalize_to_1000=True)
            return norm_x, norm_y
    return None


def _safe_image_size(image_path: str) -> Tuple[int, int]:
    """Get image dimensions, returning default 1000x1000 if read fails."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return 1000, 1000


def build_tars_prompt(obs: Dict[str, Any]) -> str:
    """Build the prompt for TARS agents from observation dictionary."""
    instruction = obs.get("query_rewritten") or obs.get("task") or ""
    sop = obs.get("step_list_rewritten")
    prompt = [
        "You are a GUI agent. You are given a task and the latest screenshot. "
        "Plan briefly, then output your action.",
        "## Output Format",
        "Thought: ...",
        "Action: ...",
        "## Action Space",
        "click(point='<point>x y</point>')",
        "long_press(point='<point>x y</point>')",
        "type(content='...')  # Append \\n to submit.",
        "scroll(point='<point>x y</point>', direction='down or up or right or left')",
        "press_home()",
        "press_back()",
        "wait() # wait for a few seconds and observe the new screen",
        "screenshot() # take a new screenshot",
        "finished(content='...')",
        "## User Instruction",
        instruction,
    ]
    if sop:
        prompt.extend(["## Your previous outputs (thoughts and actions) — screenshot is the state AFTER these; do not repeat. Decide only the NEXT action.", sop])
    return "\n".join(prompt)


__all__ = [
    "BaseGUIAgent",
    "OSAtlasAgent",
    "UITarsAgent",
    "TARS15Agent",
    "atlas_to_mobile",
    "tars_to_mobile",
]
