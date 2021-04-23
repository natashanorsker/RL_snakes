import gym
import gym_snake

# Construct Environment
env = gym.make('snake-v0')
env.grid_size = [30,30]
observation = env.reset() # Constructs an instance of the game

# Controller
game_controller = env.controller

# Grid
grid_object = game_controller.grid
grid_pixels = grid_object.grid

# Snake(s)
snakes_array = game_controller.snakes
snake_object1 = snakes_array[0]


# actions = [1,2,3,1,2,0,1,2,3,0,1]
snakes_remaining = 1
while snakes_remaining != 0:
    env.render()
    action = env.action_space.sample()#actions[i]) # take a random action
    observation, reward, done, info = env.step(action)
    snakes_remaining = info['snakes_remaining']
    print('OBS: ' , observation)
    print('Reward: ' , reward)
    print('Done: ' , done)
    print('Info: ' , info)

    env.close()