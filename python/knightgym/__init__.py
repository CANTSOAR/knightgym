from . import _rs
from .config import LIQUID, THIN, base_config
from .env import KnightGymEnv

import gymnasium as _gym

_gym.register(id="KnightGym-v0", entry_point="knightgym.env:KnightGymEnv")

__all__ = ["KnightGymEnv", "base_config", "LIQUID", "THIN", "_rs"]
from .multi import KnightGymParallel  # noqa: E402
from . import agents  # noqa: E402
