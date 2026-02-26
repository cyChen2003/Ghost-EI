import subprocess
from pathlib import Path
from PIL import Image
import shutil
from typing import Union

PathLike = Union[str, Path]


def get_screenshot(
    adb_path: str,
    out_dir: PathLike = "screenshot",
    base: str = "screenshot",
    judgement_dir: PathLike = "judgement",
) -> str:
    judgement_dir = Path(judgement_dir)
    judgement_dir.mkdir(parents=True, exist_ok=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{base}.png"
    jpg = out / f"{base}.jpg"

    # Remove stale local files first
    try:
        if png.exists(): png.unlink()
        if jpg.exists(): jpg.unlink()
    except Exception:
        pass

    # Try approach A: exec-out (fastest, avoids newline issues)
    try:
        with open(png, "wb") as f:
            res = subprocess.run(
                [adb_path, "exec-out", "screencap", "-p"],
                stdout=f, stderr=subprocess.PIPE, check=True
            )
        if not png.exists() or png.stat().st_size < 1000:
            raise RuntimeError("exec-out screencap produced empty file")
    except Exception as e:
        # Fallback approach B: store on /sdcard and pull back
        subprocess.run([adb_path, "shell", "rm", "-f", "/sdcard/__tmp_screenshot.png"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        r1 = subprocess.run([adb_path, "shell", "screencap", "-p", "/sdcard/__tmp_screenshot.png"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r1.returncode != 0:
            raise RuntimeError(f"screencap failed: {r1.stderr.decode(errors='ignore')}")
        r2 = subprocess.run([adb_path, "pull", "/sdcard/__tmp_screenshot.png", str(png)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r2.returncode != 0 or not png.exists():
            raise RuntimeError(f"adb pull failed: {r2.stderr.decode(errors='ignore')}")
        subprocess.run([adb_path, "shell", "rm", "-f", "/sdcard/__tmp_screenshot.png"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Convert to JPG
    with Image.open(png) as im:
        im.convert("RGB").save(jpg, "JPEG", quality=95)

    # Remove temporary PNG
    try:
        png.unlink()
    except Exception:
        pass

    # Save into judgement_dir following a, b, c, ... order
    # Determine current file count in judgement_dir
    existing_files = list(judgement_dir.glob("*.jpg"))
    next_letter = chr(ord('a') + len(existing_files))
    
    # If there are more than 26 files, use double-letter names (aa, ab, ac, ...)
    if len(existing_files) >= 26:
        first_letter = chr(ord('a') + (len(existing_files) // 26) - 1)
        second_letter = chr(ord('a') + (len(existing_files) % 26))
        next_filename = f"{first_letter}{second_letter}.jpg"
    else:
        next_filename = f"{next_letter}.jpg"
    
    # Copy the file into judgement_dir
    judgement_file = judgement_dir / next_filename
    shutil.copy2(jpg, judgement_file)
    
    return str(jpg)
def tap(adb_path, x, y):
    command = adb_path + f" shell input tap {x} {y}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def type(adb_path, text):
    text = text.replace("\\n", "_").replace("\n", "_")
    for char in text:
        if char == ' ':
            command = adb_path + f" shell input text %s"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char == '_':
            command = adb_path + f" shell input keyevent 66"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char.isdigit():
            command = adb_path + f" shell input text {char}"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char in '-.,!?@\'°/:;()':
            command = adb_path + f" shell input text \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)
        else:
            command = adb_path + f" shell am broadcast -a ADB_INPUT_TEXT --es msg \"{char}\""
            subprocess.run(command, capture_output=True, text=True, shell=True)


def slide(adb_path, x1, y1, x2, y2):
    command = adb_path + f" shell input swipe {x1} {y1} {x2} {y2} 500"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def back(adb_path):
    command = adb_path + f" shell input keyevent 4"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    
    
def home(adb_path):
    command = adb_path + f" shell am start -a android.intent.action.MAIN -c android.intent.category.HOME"
    subprocess.run(command, capture_output=True, text=True, shell=True)

def take_screenshot(adb_path):
    command = adb_path + ' shell screencap -p /storage/emulated/0/Pictures/1.png'
    subprocess.run(command, capture_output=True, text=True, shell=True)
