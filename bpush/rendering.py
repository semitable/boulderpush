"""
2D rendering of the Robotic's Warehouse
environment using pyglet
"""

import math
import os
import sys

import numpy as np
import math
import six
from gym import error

from bpush.environment import Direction

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite


try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(
        """
    Error occured while running `from pyglet.gl import *`
    HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'.
    If you're running on a server, you may need a virtual frame buffer; something like this should work:
    'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """
    )


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_ORANGE = (255, 165, 0)
_DARKORANGE = (255, 140, 0)
_DARKSLATEBLUE = (72, 61, 139)
_TEAL = (0, 128, 128)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK
_SHELF_COLOR = _DARKSLATEBLUE
_SHELF_REQ_COLOR = _TEAL
_AGENT_COLOR = _DARKORANGE
_AGENT_LOADED_COLOR = _RED
_AGENT_DIR_COLOR = _BLACK
_BOULDER_COLOR = (70, 70, 70)
_GOAL_COLOR = (193, 247, 198)

_SHELF_PADDING = 2


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 220
        self.icon_size = 15

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


        script_dir = os.path.dirname(__file__)

        pyglet.resource.path = [os.path.join(script_dir, "sprites")]
        pyglet.resource.reindex()

        self.img_arrow = pyglet.resource.image("arrow-right.png")
        self.img_agent = pyglet.resource.image("agent.png")

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env, return_rgb_array=False):
        glClearColor(*_BACKGROUND_COLOR, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_boulder(env)
        self._draw_agents(env)

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        # VERTICAL LINES
        for r in range(self.rows + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        0,  # LEFT X
                        (self.grid_size + 1) * r + 1,  # Y
                        (self.grid_size + 1) * self.cols,  # RIGHT X
                        (self.grid_size + 1) * r + 1,  # Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )

        # HORIZONTAL LINES
        for c in range(self.cols + 1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size + 1) * c + 1,  # X
                        0,  # BOTTOM Y
                        (self.grid_size + 1) * c + 1,  # X
                        (self.grid_size + 1) * self.rows,  # TOP Y
                    ),
                ),
                ("c3B", (*_GRID_COLOR, *_GRID_COLOR)),
            )
        batch.draw()


    def _draw_boulder(self, env):
        batch = pyglet.graphics.Batch()

        boulder = env.boulder
        dir = boulder.orientation # the push_towards direction
        size = boulder.size
        x, y = boulder.x, env.grid_size[0] - boulder.y -1
        xn, yn = x, y

        if dir in (Direction.SOUTH, Direction.NORTH):
            xn += size - 1
        else:
            y -= size - 1
        
        batch.add(
            4,
            gl.GL_QUADS,
            None,
            (
                "v2f",
                (
                    (self.grid_size + 1) * x + 1,  # TL - X
                    (self.grid_size + 1) * y + 1,  # TL - Y
                    (self.grid_size + 1) * (xn + 1),  # TR - X
                    (self.grid_size + 1) * y + 1,  # TR - Y
                    (self.grid_size + 1) * (xn + 1),  # BR - X
                    (self.grid_size + 1) * (yn + 1),  # BR - Y
                    (self.grid_size + 1) * x + 1,  # BL - X
                    (self.grid_size + 1) * (yn + 1),  # BL - Y
                ),
            ),
            ("c3B", 4 * _BOULDER_COLOR),
        )


        # draw the arrows
        arrows = []
        arrow_size = (self.grid_size + 2) / self.img_arrow.width

        for arr in range(size):
            if dir == Direction.SOUTH:
                row = boulder.y - 2
                col = boulder.x + arr
                rot = 90
            elif dir == Direction.NORTH:
                row = boulder.y + 1
                col = boulder.x + arr + 1
                rot = 270

            elif dir == Direction.EAST:
                row = boulder.y + arr
                col = boulder.x - 1
                rot = 0

            elif dir == Direction.WEST:
                row = boulder.y + arr - 1
                col = boulder.x + 2
                rot = 180

            arrows.append(
                pyglet.sprite.Sprite(
                    self.img_arrow,
                    (self.grid_size + 1) * col + arrow_size / 2,
                    self.height - (self.grid_size + 1) * (row + 1) + arrow_size / 2,
                    batch=batch,
                )
            )
        for a in arrows:
            a.update(scale=arrow_size, rotation=rot)
            a.opacity = 128

        if dir == Direction.SOUTH:
            y = yn = 0
        elif dir == Direction.NORTH:
            y = yn = env.grid_size[0]-1
        elif dir == Direction.WEST:
            x = xn = 0
        elif dir == Direction.EAST:
            x = xn = env.grid_size[1]-1
            

        batch.add(
            4,
            gl.GL_QUADS,
            None,
            (
                "v2f",
                (
                    (self.grid_size + 1) * x + 1,  # TL - X
                    (self.grid_size + 1) * y + 1,  # TL - Y
                    (self.grid_size + 1) * (xn + 1),  # TR - X
                    (self.grid_size + 1) * y + 1,  # TR - Y
                    (self.grid_size + 1) * (xn + 1),  # BR - X
                    (self.grid_size + 1) * (yn + 1),  # BR - Y
                    (self.grid_size + 1) * x + 1,  # BL - X
                    (self.grid_size + 1) * (yn + 1),  # BL - Y
                ),
            ),
            ("c3B", 4 * _GOAL_COLOR),
        )

        batch.draw()

    def _draw_agents(self, env):
        agents = []
        batch = pyglet.graphics.Batch()

        agent_size = (self.grid_size + 2) / self.img_arrow.width


        for agent in env.agents:
            row, col = agent.y, agent.x

            agents.append(
                pyglet.sprite.Sprite(
                    self.img_agent,
                    (self.grid_size + 1) * col,
                    self.height - (self.grid_size + 1) * (row + 1),
                    batch=batch,
                )
            )
        for p in agents:
            p.update(scale=0.99*agent_size)
        batch.draw()
        