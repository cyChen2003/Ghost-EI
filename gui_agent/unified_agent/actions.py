from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional


DEFAULT_SWIPE_TEMPLATES: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
    "UP": ((500, 800), (500, 200)),
    "DOWN": ((500, 200), (500, 800)),
    "LEFT": ((800, 500), (200, 500)),
    "RIGHT": ((200, 500), (800, 500)),
}


def clamp_coordinate(value: int) -> int:
    return max(0,  int(value))


def normalize_point(
    x: float,
    y: float,
    width: Optional[float] = None,
    height: Optional[float] = None,
    *,
    normalize_to_1000: bool = False,
) -> Tuple[int, int]:
    """
    Normalize raw pixel coordinates to the 0-1000 space used by MobileAgent.
    normalize_to_1000=False keeps legacy passthrough (UI-TARS behavior).
    """
    if normalize_to_1000 and width and width > 0 and height and height > 0:
        x = x / width * 1000
        y = y / height * 1000
    return clamp_coordinate(int(x)), clamp_coordinate(int(y))


@dataclass
class MobileAction:
    name: str
    args: Dict[str, object]

    def to_command(self) -> str:
        if self.name == "Tap":
            return f"Tap ({self.args['x']}, {self.args['y']})"

        if self.name == "Swipe":
            start_x, start_y = self.args["start"]
            end_x, end_y = self.args["end"]
            return f"Swipe ({start_x}, {start_y}), ({end_x}, {end_y})"

        if self.name == "Type":
            text = str(self.args.get("text", ""))
            return f"Type <text>{text}</text>"
    
        if self.name == "Open app":
            app_name = str(self.args.get("app_name", ""))
            return f"Open app ({app_name})"
        
        return self.name


def make_tap_action(x: int, y: int) -> MobileAction:
    return MobileAction(name="Tap", args={"x": clamp_coordinate(x), "y": clamp_coordinate(y)})


def make_swipe_action(direction: str) -> MobileAction:
    start_end = DEFAULT_SWIPE_TEMPLATES.get(direction.upper())
    if not start_end:
        # Default to a downward swipe if direction is unknown.
        start_end = DEFAULT_SWIPE_TEMPLATES["DOWN"]
    return MobileAction(name="Swipe", args={"start": start_end[0], "end": start_end[1]})


def make_type_action(text: str) -> MobileAction:
    return MobileAction(name="Type", args={"text": text})


def make_open_app_action(app_name: str) -> MobileAction:
    return MobileAction(name="Open app", args={"app_name": app_name.strip()})


def simple_action(name: str) -> MobileAction:
    return MobileAction(name=name, args={})
