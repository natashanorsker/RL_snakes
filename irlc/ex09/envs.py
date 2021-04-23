"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
import gym
gym.envs.register(
     id='Gambler-v0',
     entry_point='irlc.ex09old.gambler:GamblerEnv',
)

gym.envs.register(
     id='Tenv-v0',
     entry_point='irlc.ex09old.gambler:TEnv',
    max_episode_steps=100,
)

gym.envs.register(
     id='JackRental4-v0',
     entry_point='irlc.ex09old.jacks_car_rental:RentalEnv',
     max_episode_steps=1000,
     kwargs={"max_cars": 4,
             "poisson_truncation": 4,
             "cache_str": "jack_rental_environment_4"},
)

gym.envs.register(
     id='JackRental-v0',
     entry_point='irlc.ex09old.jacks_car_rental:RentalEnv',
     max_episode_steps=1000,
     kwargs={"cache_str": "jack_rental_environment"},
)  # "compress_tol": 0.01

gym.envs.register(
     id='SmallGridworld-v0',
     entry_point='irlc.gridworld.gridworld:SuttonCornerGridEnvironment',
     # max_episode_steps=100,  # Stop trying to make it happen
)

gym.envs.register( # Like MountainCar-v0, but time limit increased from 200 to 500.
    id='MountainCar500-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=500,
    reward_threshold=-110.0,
)


if __name__ == "__main__":
    print("Testing...")
    mc = gym.make('MountainCar500-v0')
    j4 = gym.make("JackRental4-v0")
    jack = gym.make("JackRental-v0")
    sg = gym.make("SmallGridworld-v0")
