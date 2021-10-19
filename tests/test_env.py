import os
import sys
import pytest
import gym
import numpy as np
from gym import spaces

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from bpush.environment import BoulderPush, Direction


def test_grid_size():
    env = BoulderPush(
        width=5,
        height=10,
        n_agents=2,
        sensor_range=3,
    )
    assert env.grid_size == (10, 5)
    env = BoulderPush(
        width=10,
        height=5,
        n_agents=2,
        sensor_range=3,
    )
    assert env.grid_size == (5, 10)


def test_action_space_0():
    env = BoulderPush(
        width=10,
        height=5,
        n_agents=2,
        sensor_range=3,
    )
    env.reset()
    assert env.action_space == spaces.Tuple(2 * (spaces.Discrete(len(Direction)), ))
    env.step(env.action_space.sample())


def test_obs_space_0():
    env = BoulderPush(
        width=10,
        height=5,
        n_agents=2,
        sensor_range=3,
    )
    env.reset()
    # assert env.observation_space == spaces.Tuple(2 * (spaces.Discrete(len(Direction)), ))

def test_push_north_0():
    env = BoulderPush(
        width=10,
        height=10,
        n_agents=2,
        sensor_range=3,
    )
    env.reset()
    env.boulder.orientation = Direction.NORTH
    env.boulder.x = 2
    env.boulder.y = 2

    env.agents[0].x = 2
    env.agents[0].y = 3
    
    env.agents[1].x = 3
    env.agents[1].y = 3

    env.grid[2, :] = 0
    env.grid[2, 0, 2:4] = -1

    env._draw_grid()

    _, rew, done, _ = env.step(2*[int(Direction.NORTH)])
    assert env.agents[0].y == 2 and env.agents[1].y == 2
    assert env.boulder.y == 1 and env.boulder.y == 1
    assert sum(rew) == 0

    _, rew, done, _ = env.step(2*[int(Direction.NORTH)])
    assert env.agents[0].y == 1 and env.agents[1].y == 1
    assert env.boulder.y == 0 and env.boulder.y == 0
    assert all([r == 1.0 for r in rew])
    assert all(done)
    
def test_push_south_0():
    env = BoulderPush(
        width=10,
        height=10,
        n_agents=2,
        sensor_range=3,
    )
    env.reset()
    env.boulder.orientation = Direction.SOUTH
    env.boulder.x = 2
    env.boulder.y = 7

    env.agents[0].x = 2
    env.agents[0].y = 6
    
    env.agents[1].x = 3
    env.agents[1].y = 6

    env.grid[2, :] = 0
    env.grid[2, 9, 2:4] = -1

    env._draw_grid()

    _, rew, done, _ = env.step(2*[int(Direction.SOUTH)])
    assert env.agents[0].y == 7 and env.agents[1].y == 7
    assert env.boulder.y == 8 and env.boulder.y == 8
    assert sum(rew) == 0

    _, rew, done, _ = env.step(2*[int(Direction.SOUTH)])
    assert env.agents[0].y == 8 and env.agents[1].y == 8
    assert env.boulder.y == 9 and env.boulder.y == 9
    assert all([r == 1.0 for r in rew])
    assert all(done)
    
def test_push_west_0():
    env = BoulderPush(
        width=10,
        height=10,
        n_agents=2,
        sensor_range=3,
    )
    env.reset()
    env.boulder.orientation = Direction.WEST
    env.boulder.x = 2
    env.boulder.y = 2

    env.agents[0].x = 3
    env.agents[0].y = 2
    
    env.agents[1].x = 3
    env.agents[1].y = 3

    env.grid[2, :] = 0
    env.grid[2, 2:4, 0] = -1
    env._draw_grid()

    _, rew, done, _ = env.step(2*[int(Direction.WEST)])
    assert env.agents[0].x == 2 and env.agents[1].x == 2
    assert env.boulder.x == 1 and env.boulder.x == 1
    assert sum(rew) == 0

    _, rew, done, _ = env.step(2*[int(Direction.WEST)])
    assert env.agents[0].x == 1 and env.agents[1].x == 1
    assert env.boulder.x == 0 and env.boulder.x == 0
    assert all([r == 1.0 for r in rew])
    assert all(done)
    
def test_push_east_0():
    env = BoulderPush(
        width=10,
        height=10,
        n_agents=2,
        sensor_range=3,
    )
    env.reset()
    env.boulder.orientation = Direction.EAST
    env.boulder.x = 7
    env.boulder.y = 2

    env.agents[0].x = 6
    env.agents[0].y = 2
    
    env.agents[1].x = 6
    env.agents[1].y = 3

    env.grid[2, :] = 0
    env.grid[2, 2:4, 9] = -1
    env._draw_grid()

    print(env.grid)
    _, rew, done, _ = env.step(2*[int(Direction.EAST)])
    assert env.agents[0].x == 7 and env.agents[1].x == 7
    assert env.boulder.x == 8 and env.boulder.x == 8
    assert sum(rew) == 0

    _, rew, done, _ = env.step(2*[int(Direction.EAST)])
    assert env.agents[0].x == 8 and env.agents[1].x == 8
    assert env.boulder.x == 9 and env.boulder.x == 9
    assert all([r == 1.0 for r in rew])
    assert all(done)
    

    

