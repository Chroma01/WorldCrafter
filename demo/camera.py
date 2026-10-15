"""Metric c2w controls: x right, y down, z forward, endpoint-exclusive frames."""

import math
import numpy as np

from worldcrafter.camera import Action, sample_chunk, relative_poses


class ControlBuffer:
    """Latest fresh keydown wins until consume; never resume an overridden key."""

    ORBIT_KEYS = {"i", "j", "k", "l"}

    KEYS = {
        "w": ("forward", 1),
        "s": ("forward", -1),
        "a": ("right", -1),
        "d": ("right", 1),
        "q": ("up", 1),
        "e": ("up", -1),
        "arrowleft": ("yaw", -1),
        "arrowright": ("yaw", 1),
        "arrowup": ("pitch", 1),
        "arrowdown": ("pitch", -1),
        # Orbit keys describe camera movement, opposite to the viewing turn.
        "j": ("yaw", 1),
        "l": ("yaw", -1),
        "i": ("pitch", -1),
        "k": ("pitch", 1),
    }

    def __init__(self):
        self.clear()
        self.speed = 2.0
        self.vertical_speed = 2.0
        self.rotation_angle = 30.0
        self.orbit_radius = 2.0

    def clear(self):
        self.held = set()
        self.selected_key = None
        self.pending_key = None

    def update(self, message):
        kind = message["type"]
        if kind == "blur":
            self.clear()
        elif kind == "key":
            key = str(message.get("key", "")).lower()
            if key not in self.KEYS:
                raise ValueError("Unsupported key")
            if message.get("down"):
                if key not in self.held:
                    self.selected_key = self.pending_key = key
                self.held.add(key)
            else:
                self.held.discard(key)
        elif kind in ("speed", "vertical_speed", "rotation_angle", "orbit_radius"):
            value = float(message["value"])
            if not math.isfinite(value):
                raise ValueError("Control setting must be finite")
            low, high = (10.0, 45.0) if kind == "rotation_angle" else (1.0, 5.0)
            setattr(self, kind, max(low, min(high, value)))
        else:
            raise ValueError("Unknown control")

    def peek(self):
        key = self.pending_key
        if key is None and self.selected_key in self.held:
            key = self.selected_key
        if key is None:
            return Action(speed=self.speed)
        field, sign = self.KEYS[key]
        value = sign * self.rotation_angle if field in ("yaw", "pitch") else sign
        speed = self.vertical_speed if field == "up" else self.speed
        orbit = key in self.ORBIT_KEYS
        return Action(
            **{field: value}, speed=speed,
            orbit=orbit, orbit_radius=self.orbit_radius if orbit else 0.0,
        ).normalized()

    def consume(self):
        action = self.peek()
        self.pending_key = None
        return action

    def has_action(self):
        return self.pending_key is not None or self.selected_key in self.held


class Camera:
    def __init__(self, world_up=None):
        self.world = np.eye(4, dtype=np.float64)
        if world_up is not None:
            from worldcrafter.gravity import gravity_frame
            self.world[:3, :3] = gravity_frame(world_up)
        self.initial_inverse = np.linalg.inv(self.world)
        self.local_chunks, self.global_chunks = [], []

    def append(self, action):
        global_pose, end = sample_chunk(self.world, action.normalized())
        global_pose = self.initial_inverse @ global_pose
        global_pose = global_pose[:, :3].astype(np.float32)
        local = relative_poses(global_pose)
        self.world = end
        self.local_chunks.append(local)
        self.global_chunks.append(global_pose)
        return self.poses()

    def poses(self):
        return np.concatenate(self.local_chunks), np.concatenate(self.global_chunks)
