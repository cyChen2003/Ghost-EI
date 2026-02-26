import os
import re
import json
import ast
import copy
import shutil
import stat
from typing import List, Dict, Any, Optional, Sequence, Union
from openai import OpenAI
from MobileAgent.api import encode_image
from MobileAgent.chat import init_judge_chat
from MobileAgent.prompt import get_judge_prompt


# --------- helpers ---------

def _guess_mime(path: str) -> str:
    p = path.lower()
    if p.endswith(".jpg") or p.endswith(".jpeg"):
        return "image/jpeg"
    if p.endswith(".png"):
        return "image/png"
    if p.endswith(".webp"):
        return "image/webp"
    if p.endswith(".bmp"):
        return "image/bmp"
    return "application/octet-stream"


def _collect_images_from_dir(images_dir: str) -> List[str]:
    """
    Collect image file paths from a directory.
    Expected names like a.png, b.jpg, c.webp ... and we sort alphabetically
    by base name (case-insensitive) so 'a' < 'b' < 'c'...
    """
    if not os.path.isdir(images_dir):
        # If it's a single file path, just return it (graceful fallback)
        return [images_dir] if os.path.isfile(images_dir) else []

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = []
    for name in os.listdir(images_dir):
        full = os.path.join(images_dir, name)
        if os.path.isfile(full) and os.path.splitext(name)[1].lower() in exts:
            files.append(full)

    # Sort by stem (filename without extension), case-insensitive
    def _stem_lower(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0].lower()

    files.sort(key=_stem_lower)
    return files


def _images_to_mm_content(
    image_input: Optional[Union[str, Sequence[str]]]
) -> List[Dict[str, Any]]:
    """
    Accepts either:
      - a directory path (str): images inside are read & sorted by name (a,b,c,d,...)
      - a single file path (str)
      - an explicit list/sequence of image file paths
    Returns OpenAI Chat Completions multi-modal parts.
    """
    if not image_input:
        return []

    # Normalize to a list of file paths
    if isinstance(image_input, str):
        image_paths = _collect_images_from_dir(image_input)
    else:
        image_paths = list(image_input)

    parts: List[Dict[str, Any]] = []
    for path in image_paths:
        if not os.path.isfile(path):
            continue
        b64 = encode_image(path)  # your provided helper
        mime = _guess_mime(path)
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })
    return parts


def _extract_obj_like(text: str) -> str:
    """Extract the first {...} block to increase parse robustness."""
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else text.strip()


def _parse_bool_dict(raw: str) -> Dict[str, bool]:
    """
    Robustly parse the model output into:
      {"task_completion": bool, "full_attack_success": bool, "partial_attack_success": bool}
    """
    if not raw:
        return {"task_completion": False, "full_attack_success": False, "partial_attack_success": False}

    s = _extract_obj_like(raw)

    # Try JSON first (normalize Python bools to JSON)
    s_json = re.sub(r"\bTrue\b", "true", s)
    s_json = re.sub(r"\bFalse\b", "false", s_json)
    try:
        obj = json.loads(s_json)
    except Exception:
        try:
            obj = ast.literal_eval(s)
        except Exception:
            return {"task_completion": False, "full_attack_success": False, "partial_attack_success": False}

    return {
        "task_completion": bool(obj.get("task_completion", False)),
        "full_attack_success": bool(obj.get("full_attack_success", False)),
        "partial_attack_success": bool(obj.get("partial_attack_success", False)),
    }


def _normalize_base_url(api_url: Optional[str]) -> Optional[str]:
    """
    Normalize a user-provided base URL:
      - None/'' -> None (use SDK default)
      - Trim whitespace/trailing slashes
      - Remove any trailing '/chat/completions' or similar path
      - Ensure it ends with '/v1' if host is api.openai.com
    """
    if not api_url:
        return None
    url = api_url.strip().rstrip("/")

    # If user mistakenly passed a full path like .../v1/chat/completions, strip it back to /v1
    for tail in ("/chat/completions", "/completions", "/responses"):
        if url.endswith("/v1" + tail):
            url = url[: -len(tail)]
            break
        if url.endswith(tail):
            # If user gave a nonstandard host but full path, strip the tail
            url = url[: -len(tail)]
            break

    # If the host is api.openai.com and doesn't end with /v1, add it
    if "api.openai.com" in url and not url.endswith("/v1"):
        url = url + "/v1"

    return url


# --------- main class ---------

class JudgeClient:
    """
    OpenAI-powered judge for MobileAgent runs.

    Usage:
        jc = JudgeClient(model="gpt-4o-mini")
        result = jc.judge(
            instruction=user_instruction_str,
            action_history=action_history_struct_or_str,
            judge_criteria=judge_criteria_struct_or_str,
            images="path/to/dir/with/a,b,c,d..."   # or single file path, or a list of paths
        )
        print(result["parsed"])  # -> {"task_completion": bool, "full_attack_success": bool, "partial_attack_success": bool}
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        api_url: Optional[str] = None,   # pass None or root like "https://api.openai.com/v1"
        max_tokens: int = 300,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        base_url = _normalize_base_url(api_url)
        # Create client with/without base_url depending on user input
        if base_url:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _build_messages(
        self,
        instruction: Any,
        action_history: Any,
        judge_criteria: Any,
        images: Optional[Union[str, Sequence[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Construct multimodal messages using your helpers."""
        op_hist = init_judge_chat()  # [["system", [{type:"text", text:...}]], ...]

        # Compose the big judging prompt using your template function
        prompt_text = get_judge_prompt(
            insctuction=instruction,  # keep your arg name
            action_history=action_history,
            judge_criteria=judge_criteria
        )

        user_content = [{"type": "text", "text": prompt_text}]
        user_content.extend(_images_to_mm_content(images))

        # Append to operation history in your structure, then convert to OpenAI's schema
        op_hist = copy.deepcopy(op_hist)
        op_hist.append(["user", user_content])

        messages = []
        for role, content in op_hist:
            messages.append({"role": role, "content": content})
        return messages

    def _complete(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        return {"raw_text": text, "openai_response": resp}

    def judge(
        self,
        instruction: Any,
        action_history: Any,
        judge_criteria: Any,
        images: Optional[Union[str, Sequence[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the judge and return:
          {
            "parsed": {"task_completion": bool, "full_attack_success": bool, "partial_attack_success": bool},
            "raw_text": "<model output>",
            "openai_response": <full OpenAI response object>
          }
        """
        messages = self._build_messages(instruction, action_history, judge_criteria, images)
        out = self._complete(messages)
        parsed = _parse_bool_dict(out["raw_text"])
        out["parsed"] = parsed
        return out

    # --------- clear judgement directory ---------

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
        
def copy_judgement_contents(
    dest_dir: str,
    src_dir: Optional[str] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:

    result = {
        "src": "",
        "dst": os.path.abspath(dest_dir),
        "files_copied": 0,
        "dirs_copied": 0,
        "bytes_copied": 0,
        "skipped": [],
        "errors": [],
        "exists": False,
    }

    # Resolve source directory; default to cwd/judgement
    if not src_dir:
        src_dir = os.path.join(os.getcwd(), "judgement")
    src_dir = os.path.abspath(src_dir)
    result["src"] = src_dir

    if os.path.basename(src_dir.rstrip(os.sep)) != "judgement":
        result["errors"].append(f"Refusing to copy from non-'judgement' path: {src_dir}")
        return result

    if not os.path.isdir(src_dir):
        return result

    result["exists"] = True


    try:
        os.makedirs(result["dst"], exist_ok=True)
    except Exception as e:
        result["errors"].append(f"Create dest failed: {result['dst']}: {e}")
        return result

    def _ensure_writable(path: str):
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        except Exception:
            pass

    try:
        with os.scandir(src_dir) as it:
            for entry in it:
                s_path = entry.path
                d_path = os.path.join(result["dst"], entry.name)

                try:
                    if entry.is_dir(follow_symlinks=False):
                        if os.path.exists(d_path):
                            if overwrite:

                                shutil.rmtree(d_path, onerror=lambda f, p, e: (_ensure_writable(p), f(p)))
                            else:
                                result["skipped"].append(d_path)
                                continue

                        shutil.copytree(s_path, d_path, dirs_exist_ok=False)
                        result["dirs_copied"] += 1


                        for root, _, files in os.walk(d_path):
                            for fn in files:
                                fp = os.path.join(root, fn)
                                try:
                                    result["bytes_copied"] += os.path.getsize(fp)
                                except Exception:
                                    pass
                    else:
                        if os.path.exists(d_path) and not overwrite:
                            result["skipped"].append(d_path)
                            continue
                        os.makedirs(os.path.dirname(d_path), exist_ok=True)
                        if os.path.exists(d_path) and overwrite:
                            _ensure_writable(d_path)
                        shutil.copy2(s_path, d_path) 
                        result["files_copied"] += 1
                        try:
                            result["bytes_copied"] += os.path.getsize(d_path)
                        except Exception:
                            pass
                except Exception as e:
                    result["errors"].append(f"{s_path} -> {d_path}: {e}")

    except Exception as e:
        result["errors"].append(f"Scan failed: {src_dir}: {e}")

    return result
