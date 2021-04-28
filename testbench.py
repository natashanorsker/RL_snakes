from rasmusmus import *


# Construct Environment
env = Snake_env()
observation = env.reset() # Constructs an instance of the game

snakes_remaining = 1
while snakes_remaining != 0:
    env.render()
    action = env.action_space.sample()#actions[i]) # take a random action
    observation, reward, done, info = env.step(action)
    snakes_remaining = info['snakes_remaining']
    # print('OBS: ' , observation)
    print(observation)
    # print('Reward: ' , reward)
    # print('Done: ' , done)
    # print('Info: ' , info)

    env.close()
