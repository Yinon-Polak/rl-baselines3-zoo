import copy
import random
from typing import Optional, Tuple

import gym
import numpy as np
from gym.core import ObsType
from gymnasium import spaces

from snake.pygame_controller import PygameController, DummyPygamController
from snake.wrappers import Direction, Point, CLOCK_WISE, CollisionType

# rgb colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20

KWARGS = {
    "w": 640,
    "h": 480,
    "use_pygame": False,
    "positive_reward": 1,
    "negative_reward": -1,
    "verbose": 0,
}


class SnakeGameAIGym(gym.Env):

    def __init__(self, **kwargs):
        super(SnakeGameAIGym, self).__init__()

        _kwargs = copy.deepcopy(KWARGS)
        _kwargs.update(kwargs)
        self.verbose = _kwargs["verbose"]
        if self.verbose > 0:
            print(f'kwargs:', _kwargs)

        self.positive_reward = _kwargs["positive_reward"]
        self.negative_reward = _kwargs["negative_reward"]
        self.w = _kwargs["w"]
        self.h = _kwargs["h"]
        self.use_pygame = _kwargs["use_pygame"]
        self.w_pixels = int(self.w / BLOCK_SIZE)
        self.h_pixels = int(self.h / BLOCK_SIZE)
        self.n_blocks = self.w_pixels * self.h_pixels
        self.screen_mat_shape = (self.h_pixels, self.w_pixels, 1)
        self.pygame_controller = PygameController(self.w, self.h,
                                                  BLOCK_SIZE) if self.use_pygame else DummyPygamController()

        # vars that are defined in reset()
        self.direction = None
        self.head = None
        self.snake = None
        self.trail = None
        self.last_trail = None
        self.score = None
        self.food = None
        self.frame_iteration = None
        self.n_games = 0
        ####

        self.pixel_color_border = np.array(BLACK)
        self.pixel_color_background = np.array(WHITE)
        self.pixel_color_body = np.array(BLUE1)
        self.pixel_color_snake_head = np.array(GREEN)
        self.pixel_color_food = np.array(RED)

        self.action_space = spaces.Discrete(3)
        # self.observation_space = spaces.Box(low=-1.0, high=1001.0, shape=(884,), dtype=np.float32)
        n_elements = self.screen_mat_shape[0] * self.screen_mat_shape[1]
        self.observation_space = spaces.Box(low=-1, high=1001, shape=(n_elements,), dtype=np.int32)

        self.reset()

    def _get_state(self):
        pixel_mat = np.zeros(self.screen_mat_shape)
        pixel_mat[:, :] = 0
        pixel_mat[self.food.get_y_x_tuple()] = 1000
        for i, body_point in enumerate(self.snake, start=1):
            pixel_mat[body_point.get_y_x_tuple()] = i

        return pixel_mat.reshape(-1)

    def step(self, action):
        return self.play_step(action)

    def render(self, mode="human"):
        pass

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.trail = []
        self.last_trail = []

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        return self._get_state(), dict()

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        self.pygame_controller.check_quit_event()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        should_early_terminate = self.frame_iteration > 100 * len(self.snake)
        if self.is_collision() or should_early_terminate:
            game_over = True
            self.n_games += 1
            reward = self.negative_reward
            info = {"score": self.score, "is_looping": len(set(self.last_trail[:100])) < 6}
            if should_early_terminate:
                info["TimeLimit.truncated"] = True

            return self._get_state(), reward, game_over, False, info

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = self.positive_reward
            self._place_food()

        else:
            crumb = self.snake.pop()
            self.trail.insert(0, crumb)
            self.last_trail.insert(0, crumb)

        # 5. update ui and clock
        self.pygame_controller.update_ui(self.food, self.score, self.snake)
        self.pygame_controller.clock_tick()
        # 6. return game over and score

        # info = {"score": self.score, "is_looping": False}
        info = {"score": self.score}
        return self._get_state(), reward, game_over, False, info

    def is_collision(self, collision_type: CollisionType = CollisionType.BOTH, pt: Point = None):
        if pt is None:
            pt = self.head
        # hits boundary
        if (collision_type == CollisionType.BORDER or collision_type == CollisionType.BOTH) and \
                (pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0):
            return True
        # hits itself
        if (collision_type == CollisionType.BODY or collision_type == CollisionType.BOTH) and \
                pt in self.snake[1:]:
            return True

        return False

    def _is_hit_itslef(self, pt=None) -> bool:
        if pt is None:
            pt = self.head

        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        if isinstance(action, np.int64) or isinstance(action, int):
            action_list = [0, 0, 0]
            action_list[action] = 1
            action = action_list
        elif not isinstance(action, list):
            raise RuntimeError(
                f"no convertor implemented for action type: {type(action)}, only np.int64 and list are supported")

        self.direction = head_to_global_direction(current_direction=self.direction, action=action)

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)


def head_to_global_direction(current_direction, action) -> Direction:
    # [straight, right, left]
    idx = CLOCK_WISE.index(current_direction)

    if np.array_equal(action, [1, 0, 0]):
        new_dir = CLOCK_WISE[idx]  # no change
    elif np.array_equal(action, [0, 1, 0]):
        next_idx = (idx + 1) % 4
        new_dir = CLOCK_WISE[next_idx]  # right turn r -> d -> l -> u
    else:  # [0, 0, 1]
        next_idx = (idx - 1) % 4
        new_dir = CLOCK_WISE[next_idx]  # left turn r -> u -> l -> d

    return new_dir
