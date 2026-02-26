import subprocess
import os
import time
import copy
import torch
import shutil
import concurrent.futures
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Callable, Optional
import base64
import openai
import shlex
from PIL import Image, ImageDraw
import openai
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
# Environment class: bundles models, OCR/detection, captioning, file management, and perception helpers
# =========================
@dataclass
class EnvConfig:
    adb_path: str
    API_url: str
    token: str
    caption_call_method: str = "api"                 # "api" | "local"
    caption_model: str = "qwen-vl-plus"              # a
    reflect_model: str = "qwen-vl-plus"           # 
    judge_model: str = "gpt-4o"
    qwen_api_key: str = ""
    add_info: str = ""
    reflection_switch: bool = True
    memory_switch: bool = True
    reasoning_switch: bool = False
    device: str = "cuda"
    temp_dir: str = "temp"
    screenshot_dir: str = "screenshot"
    file_dir: str = "./files"
    seed: int = 1234
    judgement_dir: str = "judgement"
class AndroidEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.CLEAR_ALL_LABELS = [
    "clear all", "CLEAR ALL", "Clear all",
    "全部清除", "清除全部", "全部关闭", "全部移除", "全部清理", "关闭全部", "清空全部"
    ]
        # Prepare working directories
        self._ensure_dirs()

        # Seed for reproducibility
        torch.manual_seed(cfg.seed)

        # Initialize icon caption models (local or API)
        self._init_captioner()

        # Initialize OCR / icon detection models
        self._init_perception_models()

    # ---------- Directory helpers ----------
    def _ensure_dirs(self):
        if os.path.exists(self.cfg.temp_dir):
            shutil.rmtree(self.cfg.temp_dir)
        os.makedirs(self.cfg.temp_dir, exist_ok=True)

        os.makedirs(self.cfg.screenshot_dir, exist_ok=True)
        os.makedirs(self.cfg.judgement_dir, exist_ok=True)

    # ---------- Model initialization ----------
    def _init_captioner(self):
        self.tokenizer = None
        self.local_caption_model = None

        if self.cfg.caption_call_method == "local":
            if self.cfg.caption_model == "qwen-vl-chat":
                qwen_dir = snapshot_download("qwen/Qwen-VL-Chat", revision="v1.1.0")
                self.local_caption_model = AutoModelForCausalLM.from_pretrained(
                    qwen_dir, device_map=self.cfg.device, trust_remote_code=True
                ).eval()
                self.local_caption_model.generation_config = GenerationConfig.from_pretrained(
                    qwen_dir, trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)

            elif self.cfg.caption_model == "qwen-vl-chat-int4":
                qwen_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision="v1.0.0")
                self.local_caption_model = AutoModelForCausalLM.from_pretrained(
                    qwen_dir, device_map=self.cfg.device, trust_remote_code=True, use_safetensors=True
                ).eval()
                self.local_caption_model.generation_config = GenerationConfig.from_pretrained(
                    qwen_dir, trust_remote_code=True, do_sample=False
                )
                self.tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
            else:
                raise ValueError(
                    'When using a local caption model, choose caption_model from {"qwen-vl-chat", "qwen-vl-chat-int4"}.'
                )
        elif self.cfg.caption_call_method == "api":
            # Only set the API key; inference happens during calls
            dashscope.api_key = self.cfg.qwen_api_key or ""
        else:
            raise ValueError('caption_call_method must be either "local" or "api".')

    def _init_perception_models(self):
        groundingdino_dir = snapshot_download("AI-ModelScope/GroundingDINO", revision="v1.0.0")
        self.groundingdino_model = pipeline("grounding-dino-task", model=groundingdino_dir)
        self.ocr_detection = pipeline(Tasks.ocr_detection, model="damo/cv_resnet18_ocr-detection-line-level_damo")
        self.ocr_recognition = pipeline(Tasks.ocr_recognition, model="damo/cv_convnextTiny_ocr-recognition-document_damo")

    # ---------- Utility helpers ----------
    @staticmethod
    def _list_files(folder_path: str) -> List[str]:
        return sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    @staticmethod
    def _draw_points(image_path: str, points: List[Tuple[int, int]], out_path: str) -> str:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        r = 10
        for x, y in points:
            draw.ellipse((x - r, y - r, x + r, y + r), fill="red")
        image.save(out_path)
        return out_path

    @staticmethod
    def _crop(image_path: str, box: List[int], name: str):
        img = Image.open(image_path)
        x1, y1, x2, y2 = map(int, box)
        if x1 >= x2 - 10 or y1 >= y2 - 10:
            return
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(name)

    def _generate_local_caption(self, image_file: str, query: str) -> str:
        # Only valid when running in local mode
        q = self.tokenizer.from_list_format([{"image": image_file}, {"text": query}])
        response, _ = self.local_caption_model.chat(self.tokenizer, query=q, history=None)
        return response

    def _generate_api_caption(self, image_file: str, query: str) -> str:
        """Use the OpenAI multimodal API to generate icon captions."""
        # Encode the image
        with open(image_file, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create OpenAI client
        client = openai.OpenAI(
            api_key=self.openai_api_key,  # API key provided via configuration
            base_url=self.openai_base_url  # Configurable API endpoint
        )
        # Build multimodal request
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        try:
            # Call API with simple retry logic
            for _ in range(3):
                try:
                    response = client.chat.completions.create(
                        model= self.cfg.caption_model,  # Use model from configuration
                        messages=messages,
                        max_tokens=150,  # Control output length
                        temperature=0.2  # Reduce randomness
                    )
                    return response.choices[0].message.content.strip()
                except openai.RateLimitError:
                    time.sleep(2)  # Wait between retries to respect rate limits
            
            return "Icon description unavailable"
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return "Icon description unavailable"

    def _batch_icon_describe(self, image_paths: List[str], query: str) -> Dict[int, str]:
        """
        Return a mapping of 1-based indices to captions.
        """
        icon_map: Dict[int, str] = {}
        call_local = self.cfg.caption_call_method == "local"

        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = {
                ex.submit(
                    self._generate_local_caption if call_local else self._generate_api_caption,
                    img, query
                ): idx for idx, img in enumerate(image_paths, start=1)
            }
            for fut in concurrent.futures.as_completed(futures):
                idx = futures[fut]
                try:
                    icon_map[idx] = fut.result()
                except Exception:
                    icon_map[idx] = "None"
        return icon_map

    @staticmethod
    def _merge_text_blocks(text_list: List[str], boxes: List[List[int]]) -> Tuple[List[str], List[List[int]]]:
        merged_text_blocks: List[str] = []
        merged_coordinates: List[List[int]] = []

        order = sorted(range(len(boxes)), key=lambda k: (boxes[k][1], boxes[k][0]))
        sorted_text = [text_list[i] for i in order]
        sorted_boxes = [boxes[i] for i in order]

        n = len(sorted_text)
        used = [False] * n

        for i in range(n):
            if used[i]:
                continue
            anchor = i
            group_text = [sorted_text[anchor]]
            group_boxes = [sorted_boxes[anchor]]

            for j in range(i + 1, n):
                if used[j]:
                    continue
                # Same column, adjacent rows, similar height
                if (
                    abs(sorted_boxes[anchor][0] - sorted_boxes[j][0]) < 10
                    and -10 <= sorted_boxes[j][1] - sorted_boxes[anchor][3] < 30
                    and abs(
                        (sorted_boxes[anchor][3] - sorted_boxes[anchor][1])
                        - (sorted_boxes[j][3] - sorted_boxes[j][1])
                    )
                    < 10
                ):
                    group_text.append(sorted_text[j])
                    group_boxes.append(sorted_boxes[j])
                    used[anchor] = True
                    anchor = j
                    used[anchor] = True

            merged_text = "\n".join(group_text)
            min_x1 = min(group_boxes, key=lambda x: x[0])[0]
            min_y1 = min(group_boxes, key=lambda x: x[1])[1]
            max_x2 = max(group_boxes, key=lambda x: x[2])[2]
            max_y2 = max(group_boxes, key=lambda x: x[3])[3]

            merged_text_blocks.append(merged_text)
            merged_coordinates.append([min_x1, min_y1, max_x2, max_y2])

        return merged_text_blocks, merged_coordinates

    # ---------- Perception processing ----------
    def get_perception_infos(self, screenshot_file: str) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        - Capture a fresh screenshot
        - Merge OCR text and coordinates
        - Detect icons, crop, and caption them
        - Convert bounding boxes into center coordinates
        """
        # Capture the current screen
        get_screenshot(
            self.cfg.adb_path,
            self.cfg.screenshot_dir,
            base="screenshot",
            judgement_dir=self.cfg.judgement_dir,
        )

        # Screen size
        width, height = Image.open(screenshot_file).size

        # OCR
        try:
            text, coords = ocr(screenshot_file, self.ocr_detection, self.ocr_recognition)
            text, coords = self._merge_text_blocks(text, coords)
        except Exception:
            text, coords = [], []
        # Visualize center points for debugging
        centers = [[(coordinate[0]+coordinate[2])/2, (coordinate[1]+coordinate[3])/2] for coordinate in coords]
        self._draw_points(
            screenshot_file,
            centers,
            os.path.join(self.cfg.screenshot_dir, "output_image.png"),
        )

        # Aggregate text perception information
        perception_infos: List[Dict[str, Any]] = []
        for i in range(len(coords)):
            perception_infos.append({"text": "text: " + text[i], "coordinates": coords[i]})

        # Icon detection
        icon_boxes = det(screenshot_file, "icon", self.groundingdino_model)
        for box in icon_boxes:
            perception_infos.append({"text": "icon", "coordinates": box})

        # Crop icons into temp_dir
        img_boxes: List[List[int]] = []
        img_ids: List[int] = []
        for i, info in enumerate(perception_infos):
            if info["text"] == "icon":
                img_boxes.append(info["coordinates"])
                img_ids.append(i)

        # Ensure temp_dir stores the cropped icons
        if os.path.exists(self.cfg.temp_dir):
            shutil.rmtree(self.cfg.temp_dir)
        os.makedirs(self.cfg.temp_dir, exist_ok=True)

        for i, box in zip(img_ids, img_boxes):
            out_path = os.path.join(self.cfg.temp_dir, f"{i}.jpg")
            self._crop(screenshot_file, box, out_path)

        images = self._list_files(self.cfg.temp_dir)
        if len(images) > 0:
            images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
            icon_map = {}
            prompt = 'This image is an icon from a phone screen. Please briefly describe the shape and color of this icon in one sentence.'
            if self.cfg.caption_call_method == "local":
                for i in range(len(images)):
                    image_path = os.path.join(self.cfg.temp_dir, images[i])
                    icon_width, icon_height = Image.open(image_path).size
                    if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                        des = "None"
                    else:
                        des = self._generate_local_caption(image_path, prompt)
                    icon_map[i+1] = des
            else:
                for i in range(len(images)):
                    images[i] = os.path.join(self.cfg.temp_dir, images[i])
                icon_map = self._batch_icon_describe(images, prompt)
            for i, j in zip(image_id, range(1, len(image_id)+1)):
                if icon_map.get(j):
                    perception_infos[i]['text'] = "icon: " + icon_map[j]

        for i in range(len(perception_infos)):
            perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0]+perception_infos[i]['coordinates'][2])/2), int((perception_infos[i]['coordinates'][1]+perception_infos[i]['coordinates'][3])/2)]
        
        return perception_infos, width, height

    @staticmethod
    def keyboard_present(perception_infos: List[Dict[str, Any]], height: int, ratio: float = 0.9) -> bool:
        keyboard_limit = int(ratio * height)
        for info in perception_infos:
            if info["coordinates"][1] >= keyboard_limit and "ADB Keyboard" in info["text"]:
                return True
        return False
    


    def _adb_screenshot_robust(self, out_path: str) -> bool:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        try:
            with open(out_path, "wb") as f:
                p = subprocess.run([self.cfg.adb_path, "exec-out", "screencap", "-p"],
                                stdout=f, stderr=subprocess.PIPE, check=False, text=False)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return True
        except Exception:
            pass
        tmp_remote = "/sdcard/__recents_tmp.png"
        subprocess.run([self.cfg.adb_path, "shell", "screencap", "-p", tmp_remote], check=False, text=True)
        subprocess.run([self.cfg.adb_path, "pull", tmp_remote, out_path], check=False, text=True)
        subprocess.run([self.cfg.adb_path, "shell", "rm", "-f", tmp_remote], check=False, text=True)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0

    def _find_clear_all_and_tap(self, img_path, ocr_det, ocr_rec) -> bool:
        try:
            texts, boxes = ocr(img_path, ocr_det, ocr_rec)
        except Exception:
            texts, boxes = [], []
        lowered = [t.lower() for t in texts]
        for i, t in enumerate(lowered):
            if any(label.lower() in t for label in self.CLEAR_ALL_LABELS):
                x1, y1, x2, y2 = boxes[i]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                tap(self.cfg.adb_path, cx, cy)
                return True
        return False

    def zero_background_via_ui(
        self,
        snap_path: Optional[str] = None,
        max_swipes: int = 8,
        wait: float = 0.5
    ) -> bool:
        snap_path = snap_path or os.path.join(self.cfg.screenshot_dir, "recents.jpg")
        os.makedirs(os.path.dirname(snap_path) or ".", exist_ok=True)
        
        subprocess.run([self.cfg.adb_path, "shell", "input", "keyevent", "187"], check=False, text=True)  # KEYCODE_APP_SWITCH
        time.sleep(wait)


        ok = self._adb_screenshot_robust(snap_path)
        if ok:
            with Image.open(snap_path) as im:
                W, H = im.size
        else:
            W, H = 1080, 1920  


        if ok and self._find_clear_all_and_tap(snap_path, self.ocr_detection, self.ocr_recognition):
            time.sleep(wait)
            subprocess.run([self.cfg.adb_path, "shell", "input", "keyevent", "3"], check=False, text=True)  # HOME
            return True


        y = int(0.45 * H)
        for _ in range(max_swipes):
            slide(self.cfg.adb_path, int(0.12 * W), y, int(0.85 * W), y)  # Swipe left
            time.sleep(wait)
            if self._adb_screenshot_robust(snap_path) and self._find_clear_all_and_tap(snap_path, self.ocr_detection, self.ocr_recognition):
                time.sleep(wait)
                subprocess.run([self.cfg.adb_path, "shell", "input", "keyevent", "3"], check=False, text=True)
                return True


        tap(self.cfg.adb_path, W // 2, int(0.10 * H))
        time.sleep(wait)
        subprocess.run([self.cfg.adb_path, "shell", "input", "keyevent", "3"], check=False, text=True)
        return True
    
    

    def _upload_file(self, local_path: str, remote_dir: str) -> bool:
        if not os.path.exists(local_path):
            return False

        adb = self.cfg.adb_path

        # Normalize remote directory path
        remote_dir = (remote_dir or "/sdcard/Download/").replace("\\", "/")
        if not remote_dir.endswith("/"):
            remote_dir += "/"

        # Ensure remote target root exists
        subprocess.run([adb, "shell", "mkdir", "-p", remote_dir],
                    capture_output=True, text=True, check=False)

        def _ok(proc: subprocess.CompletedProcess) -> bool:
            s = (proc.stdout or "") + "\n" + (proc.stderr or "")
            s = s.lower()
            bad = ("error:", "failed to copy", "no such file or directory", "permission denied")
            return (proc.returncode == 0) and not any(b in s for b in bad)

        def _push_file(src: str, dst_dir: str) -> bool:
            if not dst_dir.endswith("/"):
                dst_dir += "/"
            r = subprocess.run([adb, "push", src, dst_dir],
                            capture_output=True, text=True, check=False)
            return _ok(r)

        # File: push directly into remote_dir
        if os.path.isfile(local_path):
            return _push_file(local_path, remote_dir)

        # Directory: push contents into remote_dir/<root name>
        root_abs = os.path.abspath(local_path.rstrip("/\\"))
        root_base = os.path.basename(root_abs)
        top_remote = f"{remote_dir}{root_base}"

        # Create top-level directory (project root)
        subprocess.run([adb, "shell", "mkdir", "-p", top_remote],
                    capture_output=True, text=True, check=False)

        # Important: use <dir>/. so adb pushes directory contents without extra nesting
        r = subprocess.run([adb, "push", os.path.join(root_abs, "."), top_remote],
                        capture_output=True, text=True, check=False)
        if _ok(r):
            return True

        # Fallback: walk files individually (slower but more compatible)
        for dirpath, _dirnames, filenames in os.walk(root_abs):
            rel = os.path.relpath(dirpath, root_abs).replace("\\", "/")
            dst_dir = top_remote if rel == "." else f"{top_remote}/{rel}"
            subprocess.run([adb, "shell", "mkdir", "-p", dst_dir],
                        capture_output=True, text=True, check=False)
            for fn in filenames:
                src = os.path.join(dirpath, fn)
                if not _push_file(src, dst_dir):
                    return False
        return True


    def upload_files(
        self,
        local_dir: Optional[str] = None,
        files_name: Optional[List[str]] = None,
        remote_dir: str = "/sdcard/Download/",
        paths: Optional[List[str]] = None
    ) -> bool:
        """
        - Items without an extension typically represent directories (for example, "proj"), while items with an extension typically represent files (such as "a.txt").
        In practice we still rely on the file system (isfile/isdir) to avoid misclassification.
        """
        candidate_paths: List[str] = []

        if paths:
            candidate_paths = list(paths)
        elif files_name:
            for name in files_name:
                p = name if (not local_dir) or os.path.isabs(name) else os.path.join(local_dir, name)
                candidate_paths.append(p)
        else:
            return False

        # Deduplicate while preserving order to avoid repeated pushes
        seen = set()
        uniq_paths: List[str] = []
        for p in candidate_paths:
            q = os.path.abspath(p)
            if q not in seen:
                uniq_paths.append(p)
                seen.add(q)

        for p in uniq_paths:
            if not os.path.exists(p):
                return False
            if not self._upload_file(p, remote_dir):
                return False
        return True


    def reset_env(self, remote_dir: str = "/sdcard/Download/") -> bool:
        """
        A more robust way to clear everything under remote_dir (including hidden files and subdirectories) and avoid leftover empty folders.
        Implementation: adb shell 'find <dir> -mindepth 1 -exec rm -rf {} \;'
        """
        adb = self.cfg.adb_path
        print(f"the adb path is {adb}")
        subprocess.run([adb, "shell", "rm", "-rf", remote_dir.rstrip("/") + "/*"], check=False, text=True)
        return True
    
    def open_files(self) -> bool:
        subprocess.run([self.cfg.adb_path, "shell", "am", "start", "-a", "android.intent.action.VIEW", 
                        "-d", "content://com.android.externalstorage.documents/document/primary%3ADownload",
                        "-t", "vnd.android.document/directory"
                        ], check=False, text=True)
        return True
