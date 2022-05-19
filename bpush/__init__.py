import gym
from gym.envs.registration import register
# from bpush import BoulderPush, RewardType, Action
import itertools


_sizes = {
    "tiny": (5, 5),
    "small": (8, 8),
    "medium": (12, 12),
    "large": (20, 20),
}

_penalties = {
    "easy": 0.01,
    "medium": 0.05,
    "hard": 0.1,
    "vhard": 0.2,
}

for size in _sizes.keys():
    for n_agents in range(1, 5):
        for penalty in _penalties.keys():
            register(
            id=f"bpush-{size}-{n_agents}ag-{penalty}-v0",
            entry_point="bpush.environment:BoulderPush",
            kwargs={
                "height": _sizes[size][0],
                "width": _sizes[size][1],
                "n_agents": n_agents,
                "sensor_range": 4,
                "penalty": _penalties[penalty]
            },
)
