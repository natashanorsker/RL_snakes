import numpy as np
from gym_snake.envs.snake import Snake
from gym_snake.envs.snake_env import SnakeEnv
from gym_snake.envs.snake.controller import Controller
from gym_snake.envs.snake.grid import Grid

class Mod_Grid(Grid):
    def __init__(self, grid_size=[30, 30], unit_size=10, unit_gap=1):
        super().__init__(grid_size=grid_size, unit_size=unit_size, unit_gap=unit_gap)


    def place_food(self, coord):
        """
        Draws a food at the coord. Ensures the same placement for
        each food at the beginning of a new episode. This is useful for
        experimentation with curiosity driven behaviors.

        num - the integer denoting the
        """
        if self.open_space < 1 or not np.array_equal(self.color_of(coord), self.SPACE_COLOR):
            return False
        self.draw(coord, self.FOOD_COLOR)
        self.food_coord = coord
        return True


    def new_food(self):
        """
        Draws a food on a random, open unit of the grid.
        Returns true if space left. Otherwise returns false.
        """

        if self.open_space < 1:
            return False
        coord_not_found = True
        while (coord_not_found):
            coord = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
            if np.array_equal(self.color_of(coord), self.SPACE_COLOR):
                coord_not_found = False
        self.draw(coord, self.FOOD_COLOR)
        self.food_coord = coord
        return True

class Mod_Controller(Controller):
    def __init__(self, grid_size=[30, 30], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1,
                 random_init=True):

        assert n_snakes < grid_size[0] // 3
        assert n_snakes < 25
        assert snake_size < grid_size[1] // 2
        assert unit_gap >= 0 and unit_gap < unit_size

        self.snakes_remaining = n_snakes
        self.grid = Mod_Grid(grid_size, unit_size, unit_gap)
        self.dir_array = np.array([[0,1],[1,0],[0,-1],[-1,0]])

        self.snakes = []
        self.dead_snakes = []
        for i in range(1, n_snakes + 1):
            start_coord = [i * grid_size[0] // (n_snakes + 1), snake_size + 1]
            self.snakes.append(Snake(start_coord, snake_size))
            color = [self.grid.HEAD_COLOR[0], i * 10, 0]
            self.snakes[-1].head_color = color
            self.grid.draw_snake(self.snakes[-1], color)
            self.dead_snakes.append(None)

        if not random_init:
            for i in range(2, n_foods + 2):
                start_coord = [i * grid_size[0] // (n_foods + 3), grid_size[1] - 5]
                self.grid.place_food(start_coord)
        else:
            for i in range(n_foods):
                self.grid.new_food()

    def get_food_dir(self):
        head = self.snakes[0].head
        food = np.array(self.grid.food_coord)
        dir_vector = food - head
        if dir_vector[0] < 0:
            if dir_vector[1] > 0:
                return 7
            elif dir_vector[1] < 0:
                return 5
            else:
                return 6
        elif dir_vector[0] > 0:
            if dir_vector[1] < 0:
                return 3
            elif dir_vector[1] > 0:
                return 1
            else:
                return 2
        elif dir_vector[1] > 0:
            return 0
        else:
            return 4

    def obs(self):
        obs_list = [-1] * 6
        if self.snakes[0] == None:
            return tuple(obs_list)
        head = self.snakes[0].head
        obs_list[4] = self.snakes[0].direction
        obs_list[5] = self.get_food_dir()
        for i in range(4):
            coord = head + self.dir_array[i]
            if self.grid.off_grid(coord) or np.any([np.all(coord == arr) for arr in self.snakes[0].body]):
                obs_list[i] = 1
            elif self.grid.food_space(coord):
                obs_list[i] = 2
            else:
                obs_list[i] = 0
        return tuple(obs_list)


    def move_result(self, direction, snake_idx=0):
        """
        Checks for food and death collisions after moving snake. Draws head of snake if
        no death scenarios.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return 0

        # Check for death of snake
        if self.grid.check_death(snake.head):
            self.dead_snakes[snake_idx] = self.snakes[snake_idx]
            self.snakes[snake_idx] = None
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            self.grid.connect(snake.body.popleft(), snake.body[0], self.grid.SPACE_COLOR)
            reward = -1
        # Check for reward
        elif self.grid.food_space(snake.head):
            self.grid.draw(snake.body[0], self.grid.BODY_COLOR) # Redraw tail
            self.grid.connect(snake.body[0], snake.body[1], self.grid.BODY_COLOR)
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            reward = 10
            self.grid.new_food()
        else:
            reward = -0.1
            empty_coord = snake.body.popleft()
            self.grid.connect(empty_coord, snake.body[0], self.grid.SPACE_COLOR)
            self.grid.draw(snake.head, snake.head_color)

        self.grid.connect(snake.body[-1], snake.head, self.grid.BODY_COLOR)

        return reward


    def step(self, directions):
        """
        Takes an action for each snake in the specified direction and collects their rewards
        and dones.

        directions - tuple, list, or ndarray of directions corresponding to each snake.
        """

        # Ensure no more play until reset
        if self.snakes_remaining < 1 or self.grid.open_space < 1:
            if type(directions) == type(int()) or len(directions) == 1:
                return self.obs(), 0, True, {"snakes_remaining": self.snakes_remaining}
            else:
                return self.obs(), [0] * len(directions), True, {"snakes_remaining": self.snakes_remaining}

        rewards = []

        if type(directions) == type(int()):
            directions = [directions]

        for i, direction in enumerate(directions):
            if self.snakes[i] is None and self.dead_snakes[i] is not None:
                self.kill_snake(i)
            self.move_snake(direction, i)
            rewards.append(self.move_result(direction, i))

        done = self.snakes_remaining < 1 or self.grid.open_space < 1
        if len(rewards) == 1:
            return self.obs(), rewards[0], done, {"snakes_remaining": self.snakes_remaining}
        else:
            return self.obs(), rewards, done, {"snakes_remaining": self.snakes_remaining}

class Snake_env(SnakeEnv):
    def __init__(self, grid_size):
        super().__init__(grid_size=grid_size)
    # def reset(self):

    def step(self, action):
        self.obs, rewards, done, info = self.controller.step(action)
        self.last_obs = self.controller.grid.grid.copy()
        return self.obs, rewards, done, info


    def reset(self):
        self.controller = Mod_Controller(self.grid_size, self.unit_size, self.unit_gap, self.snake_size, self.n_snakes, self.n_foods, random_init=self.random_init)
        self.last_obs = self.controller.grid.grid.copy()
        return  self.controller.obs()


        



