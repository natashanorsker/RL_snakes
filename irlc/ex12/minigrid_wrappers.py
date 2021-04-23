"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from gym_minigrid.wrappers import ImgObsWrapper
import gym
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

class ProjectObservationSpaceWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """
    def __init__(self, env, dims):
        super().__init__(env)
        """
        If s is the n x n x 3 tensor of observations, we want to reduce it to 
        s[:,:,dims]. To do so we have to manipulate the bounding-box object (type gym.spaces.Box)
        If box is the bounding box object, then:
        box.low[i,j,k] <= s[i,j,k] <= box.high[i,j,k]
        i.e. we also have to reduce the dimension of the upper/lower bounds.        
        """
        box = self.observation_space.spaces['image'] # Get bounding box
        """ If you insert a breakpoint, you will notice the upper bound is not set correctly for some reason.
        We have to reduce this to help the linear tile encoder. 
        nbounds (below) include the ocrrect bounds such that s[i,j,k] <= nbounds[k] for all i, j
        """
        nbounds = [max(OBJECT_TO_IDX.values()), max(COLOR_TO_IDX.values()), 3]
        for i in range(box.high.shape[2]):
            # TODO: 1 lines missing.
            raise NotImplementedError("update box.high to the maximum value as defined in nbounds")
        """
        Now reduce the size of box.high, box.low to match the desired size of the bounding box. 
        """
        # TODO: 2 lines missing.
        raise NotImplementedError("Reduce dimensionality of box.low, box.high similar to s above.")
        self.observation_space.spaces['image'] = box
        self.dims = dims

    def observation(self, obs):
        """
        Only retains self.dims of s=obs['image']. I.e. output should have dimension
        n x n x len(dims). See above for hints.
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("update x = obs['image'] to remove unwanted dimensions according to self.dims")
        obs['image'] = x
        return obs


class HashableImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env,dims=None):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs): 
        """
        Return obs['image'] as a flat tuple, i.e. (obs['image'][0,0,0], obs['image'][1,0,0],...)
        note order is irrelevant since we will use it for tabular learning.
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")

exp = f"experiments/cliffwalk_semigrad_Sarsa"

if __name__ == "__main__":
    env0 = gym.make("MiniGrid-Empty-5x5-v0")
    # Part 1: Linear encoding
    s = env0.reset()
    print("Raw environment, s, has keys: ")
    print(s.keys())
    print("Image has shape:")
    print(s['image'].shape)
    print("Content of image:")
    for d in range(3):
        print(f"d={d}\n", s['image'][:,:,d])

    print("Striping two dimensions from the image (note result is randomized due to different start positions):") 
    env0 = ProjectObservationSpaceWrapper(env0, dims=(0,2))
    s = env0.reset()
    print(s['image'].shape)
    print(s['image'][:,:,0],"\n", s['image'][:,:,0])
    print("Size of observation space (lower bound)")
    print(env0.observation_space.spaces['image'].low.shape)
    env = ImgObsWrapper(env0)
    print("New boundaries are now (all values are the same):")
    print(env.observation_space.low.shape, env.observation_space.low[0,0,0] )
    print(env.observation_space.high.shape, env.observation_space.high[0,0,0] ) 

    # Part 2: Hashable encoding for tabular learning
    envh = HashableImgObsWrapper(env0)  
    print("Resetting and printing the hashable environment:")
    s = envh.reset()
    print("Length of observation: ")
    print(len(s))
    print("First elements of observation is s=", s[:10]) 
