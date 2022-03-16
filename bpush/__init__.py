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

for size in _sizes.keys():
    for n_agents in range(1, 10):

        register(
        id=f"bpush-{size}-{n_agents}ag-v0",
        entry_point="bpush.environment:BoulderPush",
        kwargs={
            "height": _sizes[size][0],
            "width": _sizes[size][1],
            "n_agents": n_agents,
            "sensor_range": 4,
        },
)