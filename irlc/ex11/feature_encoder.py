"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
from math import floor
from gym.spaces.box import Box
import numpy as np

class FeatureEncoder():
    def __init__(self, env):
        self.env = env
        self.w = np.zeros((self.d, ))

    @property
    def d(self):
        raise NotImplementedError()

    def x(self, s, a):
        raise NotImplementedError()

    def get_Qs(self, state):
        actions = list( self.env.P[state].keys() if hasattr(self.env, 'P') else range(self.env.action_space.n) )
        Qs = [self(state, a) for a in actions]
        return tuple(actions), tuple(Qs)

    def get_optimal_action(self, state):
        actions, Qa = self.get_Qs(state)
        if len(actions) == 0:
            print("Bad actions list")
        a_ = np.argmax(np.asarray(Qa) + np.random.rand(len(Qa)) * 1e-8)
        return actions[a_]

    def __call__(self, s, a):
        return self.x(s, a) @ self.w

    def __getitem__(self, item):
        s,a = item
        return self.__call__(s, a)


class GridworldXYEncoder(FeatureEncoder):
    def __init__(self, env):
        self.env = env
        self.na = self.env.action_space.n
        self.ns = 2
        super().__init__(env)

    @property
    def d(self):
        return self.na*self.ns

    def x(self, s, a):
        x,y = s
        xx = [np.zeros(self.ns) for _ in range(self.na)]
        xx[a][0] = x
        xx[a][1] = y
        # return xx[a]
        xx = np.concatenate(xx)
        return xx

class SimplePacmanExtractor(FeatureEncoder):
    def __init__(self, env):
        self.env = env
        from irlc.pacman.featureExtractors import SimpleExtractor
        # from reinforcement.featureExtractors import SimpleExtractor
        self._extractor = SimpleExtractor()
        self.fields = ["bias", "#-of-ghosts-1-step-away", "#-of-ghosts-1-step-away", "eats-food", "closest-food"]
        super().__init__(env)

    def x(self, s, a):
        xx = np.zeros_like(self.w)
        ap = self.env._actions_gym2pac[a]
        for k, v in self._extractor.getFeatures(s, ap).items():
            xx[self.fields.index(k)] = v
        return xx

    @property
    def d(self):
        return len(self.fields)

class LinearQEncoder(FeatureEncoder):

    @property
    def d(self):
        return self.max_size

    def __init__(self, env, tilings=8, max_size=2048):

        if isinstance(env.observation_space, Box):
            # use linear tile encoding. Requires num_of_tilings and max_size to be set
            # os = env.unwrapped.observation_space
            # This might be a problem: In minigrid, we cannot use the unwrapped space
            os = env.observation_space
            low = os.low
            high = os.high
            scale = tilings / (high - low)
            hash_table = IHT(max_size)
            self.max_size = max_size
            def tile_representation(s, action):
                s_ = list( (s*scale).flat )
                active_tiles = tiles(hash_table, tilings, s_, [action]) # (s * scale).tolist()
                # if 0 not in active_tiles:
                #     active_tiles.append(0)
                return active_tiles
            self.get_active_tiles = tile_representation
        else:
            # raise Exception("Implement in new class")
            """ 
            Use Fixed Sparse Representation. See: 
            https://castlelab.princeton.edu/html/ORF544/Readings/Geramifard%20-%20Tutorial%20on%20linear%20function%20approximations%20for%20dynamic%20programming%20and%20RL.pdf            
            """
            ospace = env.observation_space
            simple = False
            if not isinstance(ospace, tuple):
                ospace = (ospace,)
                simple = True

            sz = []
            for j,disc in enumerate(ospace):
                sz.append( disc.n )

            total_size = sum(sz)
            csum = np.cumsum(sz,) - sz[0]
            self.max_size = total_size * env.action_space.n

            def fixed_sparse_representation(s, action):
                if simple:
                   s = (s,)
                s_encoded = [cs + ds + total_size * action for ds,cs in zip(s, csum)]
                return s_encoded
            self.get_active_tiles = fixed_sparse_representation
        super().__init__(env)

    def x(self, s, a):
        x = np.zeros(self.d)
        at = self.get_active_tiles(s, a)
        x[at] = 1.0
        return x

"""
Following code contains the tile-coding utilities copied from:
http://incompleteideas.net/tiles/tiles3.py-remove
"""
class IHT:
    "Structure to handle collisions"

    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0: print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


def hash_coords(coordinates, m, read_only=False):
    if isinstance(m, IHT): return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int): return hash(tuple(coordinates)) % m
    if m is None: return coordinates


def tiles(iht_or_size, num_tilings, floats, ints=None, read_only=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    if ints is None:
        ints = []
    qfloats = [floor(f * num_tilings) for f in floats]
    tiles = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        tiles.append(hash_coords(coords, iht_or_size, read_only))
    return tiles
