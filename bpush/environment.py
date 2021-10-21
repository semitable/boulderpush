import logging

from collections import defaultdict, OrderedDict
import gym
from gym import spaces

from bpush.utils import MultiAgentActionSpace, MultiAgentObservationSpace

import random
from enum import IntEnum
import numpy as np

from typing import List, Tuple, Optional, Dict


_AXIS_Z = 0
_AXIS_Y = 1
_AXIS_X = 2

_LAYER_AGENTS = 0
_LAYER_BOULDER = 1
_LAYER_GOAL = 2


class _VectorWriter:
    def __init__(self, size: int):
        self.vector = np.zeros(size, dtype=np.float32)
        self.idx = 0

    def write(self, data):
        data_size = len(data)
        self.vector[self.idx : self.idx + data_size] = data
        self.idx += data_size

    def skip(self, bits):
        self.idx += bits


class Direction(IntEnum):
    NORTH = 0
    SOUTH = 1
    WEST = 2
    EAST = 3


class Entity:
    def __init__(self, id_: int, x: int, y: int):
        self.id = id_
        self.prev_x = None
        self.prev_y = None
        self.x = x
        self.y = y


class Agent(Entity):
    counter = 0

    def __init__(self, x: int, y: int):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)


class Boulder(Entity):
    counter = 0

    def __init__(self, x, y, size, orientation):
        Boulder.counter += 1
        super().__init__(Boulder.counter, x, y)
        self.size = size
        self.orientation = orientation


class BoulderPush(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        height: int,
        width: int,
        n_agents: int,
        sensor_range: int,
    ):
        """The boulder-push environment

        :param height: The height (max y-coordinate) of the grid-world
        :type height: int
        :param width: The width (max x-coordinate) of the grid-world
        :type width: int
        :param n_agents: The number of agents (and also the size of the boulder)
        :type n_agents: int
        :param sensor_range: The range of perception of the agents
        :type sensor_range: int
        """

        self.grid_size = (height, width)

        self.failed_pushing_penalty = 0.001

        self.n_agents = n_agents
        self.sensor_range = sensor_range
        self.reward_range = (0, 1)

        self._cur_steps = 0

        self.grid = np.zeros((3, *self.grid_size), dtype=np.int32)

        sa_action_space = spaces.Discrete(len(Direction))
        self.action_space = spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.agents: List[Agent] = []

        self.goals = None
        self.boulder = None

        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2
        self._obs_length = len(Direction) + 2 * self._obs_sensor_locations

        self.observation_space = spaces.Tuple(
            tuple(
                [
                    spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=(self._obs_length,),
                        dtype=np.float32,
                    )
                    for _ in range(n_agents)
                ]
            )
        )

        self.renderer = None

    def _make_obs(self, agent):

        y_scale, x_scale = self.grid_size[0] - 1, self.grid_size[1] - 1

        min_x = agent.x - self.sensor_range
        max_x = agent.x + self.sensor_range + 1

        min_y = agent.y - self.sensor_range
        max_y = agent.y + self.sensor_range + 1
        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[1])
            or (max_y > self.grid_size[0])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_BOULDER], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_BOULDER]

        agents = padded_agents[min_y:max_y, min_x:max_x].reshape(-1)
        boulder = padded_shelfs[min_y:max_y, min_x:max_x].reshape(-1)

        obs = _VectorWriter(self.observation_space[agent.id - 1].shape[0])

        obs.write(np.eye(4)[int(self.boulder.orientation)])
        obs.write(agents)
        obs.write(boulder)

        return obs.vector

    def reset(self):
        Boulder.counter = 0
        Agent.counter = 0
        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # n_xshelf = (self.grid_size[1] - 1) // 3
        # n_yshelf = (self.grid_size[0] - 2) // 9
        self.grid[:] = 0

        # Spawn a boulder..
        # First choose an orientation (pushing boulder north, south, east, or west)
        push_towards = Direction(random.randint(0, 3))

        if push_towards == Direction.SOUTH or push_towards == Direction.NORTH:
            x = random.randint(0, self.grid_size[1] - self.n_agents)
            y = random.randint(1, self.grid_size[0] - 2)
            self.grid[_LAYER_BOULDER, y, x : x + self.n_agents] = 1

        else:
            x = random.randint(1, self.grid_size[1] - 2)
            y = random.randint(0, self.grid_size[0] - self.n_agents)
            self.grid[_LAYER_BOULDER, y : y + self.n_agents, x] = 1

        # set goal
        if push_towards == Direction.SOUTH:
            self.grid[_LAYER_GOAL, self.grid_size[0] - 1, x : x + self.n_agents] = -1
        elif push_towards == Direction.NORTH:
            self.grid[_LAYER_GOAL, 0, x : x + self.n_agents] = -1
        elif push_towards == Direction.EAST:
            self.grid[_LAYER_GOAL, y : y + self.n_agents, self.grid_size[0] - 1] = -1
        elif push_towards == Direction.WEST:
            self.grid[_LAYER_GOAL, y : y + self.n_agents, 0] = -1

        # spawn the boulder
        self.boulder = Boulder(x, y, self.n_agents, push_towards)

        self.agents = []
        for _ in range(self.n_agents):
            while True:
                x, y = random.randint(0, self.grid_size[1] - 1), random.randint(
                    0, self.grid_size[0] - 1
                )
                if (
                    self.grid[_LAYER_AGENTS, y, x] == 0
                    and self.grid[_LAYER_BOULDER, y, x] == 0
                ):
                    self.agents.append(Agent(x, y))
                    self.grid[_LAYER_AGENTS, y, x] = 1
                    break

        return tuple([self._make_obs(agent) for agent in self.agents])

    def _draw_grid(self):
        self.grid[_LAYER_AGENTS, :] = 0
        self.grid[_LAYER_BOULDER, :] = 0

        if self.boulder.orientation in (Direction.SOUTH, Direction.NORTH):
            self.grid[
                _LAYER_BOULDER,
                self.boulder.y,
                self.boulder.x : self.boulder.x + self.n_agents,
            ] = 1
        else:
            self.grid[
                _LAYER_BOULDER,
                self.boulder.y : self.boulder.y + self.n_agents,
                self.boulder.x,
            ] = 1
        for agent in self.agents:
            self.grid[_LAYER_AGENTS, agent.y, agent.x] = 1

    def step(
        self, actions: List[Direction]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        assert len(actions) == len(self.agents)

        done = False
        reward = np.zeros(self.n_agents, np.float32)
        # first check if the agents manage to push the boulder
        if (
            self.boulder.orientation == Direction.NORTH
            and self.grid[
                _LAYER_AGENTS,
                self.boulder.y + 1,
                self.boulder.x : self.boulder.x + self.n_agents,
            ].sum()
            == self.n_agents
            and all([a == Direction.NORTH for a in actions])
        ):
            # pushing boulder north
            self.boulder.y -= 1
            for agent in self.agents:
                agent.y -= 1
            self._draw_grid()
            done = not self.grid[_LAYER_BOULDER:].sum(axis=0).any()

        elif (
            self.boulder.orientation == Direction.SOUTH
            and self.grid[
                _LAYER_AGENTS,
                self.boulder.y - 1,
                self.boulder.x : self.boulder.x + self.n_agents,
            ].sum()
            == self.n_agents
            and all([a == Direction.SOUTH for a in actions])
        ):
            # pushing boulder south
            self.boulder.y += 1
            for agent in self.agents:
                agent.y += 1
            self._draw_grid()
            done = not self.grid[_LAYER_BOULDER:].sum(axis=0).any()

        elif (
            self.boulder.orientation == Direction.EAST
            and self.grid[
                _LAYER_AGENTS,
                self.boulder.y : self.boulder.y + self.n_agents,
                self.boulder.x - 1,
            ].sum()
            == self.n_agents
            and all([a == Direction.EAST for a in actions])
        ):
            # pushing boulder east
            self.boulder.x += 1
            for agent in self.agents:
                agent.x += 1
            self._draw_grid()
            done = not self.grid[_LAYER_BOULDER:].sum(axis=0).any()

        elif (
            self.boulder.orientation == Direction.WEST
            and self.grid[
                _LAYER_AGENTS,
                self.boulder.y : self.boulder.y + self.n_agents,
                self.boulder.x + 1,
            ].sum()
            == self.n_agents
            and all([a == Direction.WEST for a in actions])
        ):
            # pushing boulder west
            self.boulder.x -= 1
            for agent in self.agents:
                agent.x -= 1
            self._draw_grid()
            done = not self.grid[_LAYER_BOULDER:].sum(axis=0).any()

        else:
            # just move agents around
            # we will use classical MARL approach for resolving collisions:
            # in a random order, commit each agents move before moving to the next.
            # later agents (in the shuffled order) will be in a disadvantage
            for idx in sorted(range(self.n_agents), key=lambda _: random.random()):
                action = actions[idx]
                agent = self.agents[idx]

                self.grid[_LAYER_AGENTS, agent.y, agent.x] = 0
                if (
                    action == Direction.NORTH
                    and agent.y > 0
                    and self.grid[_LAYER_AGENTS, agent.y - 1, agent.x]
                    == 0
                ):
                    if self.grid[_LAYER_BOULDER, agent.y - 1, agent.x] == 0:
                        agent.y -= 1
                    else:
                        reward[idx] -= self.failed_pushing_penalty
                elif (
                    action == Direction.SOUTH
                    and agent.y < self.grid_size[0] - 1
                    and self.grid[_LAYER_AGENTS, agent.y + 1, agent.x] == 0
                ):
                    if self.grid[_LAYER_BOULDER, agent.y + 1, agent.x] == 0:
                        agent.y += 1
                    else:
                        reward[idx] -= self.failed_pushing_penalty
                elif (
                    action == Direction.WEST
                    and agent.x > 0
                    and self.grid[_LAYER_AGENTS, agent.y, agent.x - 1] == 0
                ):
                    if self.grid[_LAYER_BOULDER, agent.y, agent.x - 1] == 0:
                        agent.x -= 1
                    else:
                        reward[idx] -= self.failed_pushing_penalty

                elif (
                    action == Direction.EAST
                    and agent.x < self.grid_size[0] - 1
                    and self.grid[_LAYER_AGENTS, agent.y, agent.x + 1] == 0
                ):
                    if self.grid[_LAYER_BOULDER, agent.y, agent.x + 1] == 0:
                        agent.x += 1
                    else:
                        reward[idx] -= self.failed_pushing_penalty
                
                self._draw_grid()

        new_obs = tuple([self._make_obs(agent) for agent in self.agents])
        reward += 1.0 if done else 0.0
        info = {}
        return (
            new_obs,
            reward,
            self.n_agents * [done],
            info,
        )

    def render(self, mode="human"):
        if not self.renderer:
            from bpush.rendering import Viewer

            self.renderer = Viewer(self.grid_size)
        return self.renderer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        ...


if __name__ == "__main__":
    env = BoulderPush(8, 8, 3, 3)
    env.reset()
    import time
    from tqdm import tqdm

    time.sleep(2)
    # env.render()
    # env.step(18 * [Action.LOAD] + 2 * [Action.NOOP])

    for _ in tqdm(range(1000000)):
        # time.sleep(2)
        # env.render()
        actions = env.action_space.sample()
        env.step(actions)
