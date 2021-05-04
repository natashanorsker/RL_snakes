from rasmusmus import *


grid_size = [10, 10]
# Construct Environment
env = Snake_env(grid_size=grid_size)
observation = env.reset() # Constructs an instance of the game
while True:
    snakes_remaining = 1
    env.reset()
    while snakes_remaining != 0:
        env.render()
        if env.controller.snakes[0] is not None:
            headcoord = env.controller.snakes[0].head
        action = 0#actions[i]) # take a random action
        observation, reward, done, info = env.step(action)
        snakes_remaining = info['snakes_remaining']
        # print('OBS: ' , observation)
        print(observation)

        # print('Reward: ' , reward)
        # print('Done: ' , done)
        # print('Info: ' , info)

        env.close()
