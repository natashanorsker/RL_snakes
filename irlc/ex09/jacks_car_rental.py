"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf). 
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import poisson
from irlc.ex09.mdp import MDP
from irlc.ex09.value_iteration import value_iteration
from irlc import savepdf
from irlc.utils.timer import Timer

def tpoiss2(lamb, k_max):
    '''
    Given k_max and lambda, and defining the poisson PDF as:
        p_k = Poisson(k|lambda=lamb)

    the function computes:
        [ (0,p_0), (1, p_1), (k_max-1, p_{kmax-1}), (kmax, 1-sum_{k=0}^{k_max-1} p_k) ]

    I.e. the last element contains the tail of the poisson PDF, and the probabilities will sum to 1.
    This is useful in the car-environment since we know that once the poisson-events (rental and returns)
    exceed some number the excess quantity is ignored by the problem, hence only the probability
    of exceeding the quantity is useful (the last element).
    '''
    ks = list(range(k_max ))
    pk = poisson.pmf(k=ks, mu=lamb)
    pk = pk.tolist() + [ 1-sum(pk)]
    ks = ks + [k_max]
    return zip(ks, pk)

class JackRentalMDP(MDP):
    def __init__(self, max_cars=20, **kwargs):
        self.max_cars = max_cars # max cars in any one dealership
        self.max_move = 5 # Max cars we are allowed to move.
        # Important: States will be of the form s=(c1, c2) which are the cars in the two rental locations.
        initial_state = (self.max_move//2,self.max_move//2)

        self.move_cost = -2
        self.park_cost = -4 # possible something to implement later for the extended problem. not used now.

        self.car_rent = 10   # Cost of renting the car
        self.demand_rate_1 = 3 # Poisson-rates for returns/rentals; see problem description.
        self.demand_rate_2 = 4
        self.return_rate_1 = 3
        self.return_rate_2 = 2
        self.ncol, self.nrow = max_cars+1, max_cars+1
        super().__init__(initial_state=initial_state, **kwargs)

    def A(self, s):
        """ Recall s = (cars at location 1, cars at location 2)
        and the actions are the number of cars to move from location 1 to location 2.
        (it can be negative to indicate moves in the other direction). The key restriction is that
        you cannot more more cars than there are at any one location (but there is no limit on excess number of cars).
        """
        c1, c2 = s[0], s[1]
        # TODO: 3 lines missing.
        raise NotImplementedError("")
        return a

    def overnight_in_out(self, cars, demand_rate, return_rate):
        """
        There are many ways to solve this problem, but I found this helper method useful:
        For a single car dealership, this returns the effect of an "overnight update" by taking

        > c2: Number of cars in dealership
        > demand_rate: Rate of demand in dealership
        > return_rate: rate of return of cars in dealership

        then compute the effect of cars being rented and returned. In other words, it returns a dictionary of the form:

        > d = { (n1, r1): p1, (n2, r2): p2, ...}

        Where (n1, r1) is a possible (new) number of cars, and associated reward, of overnight rentals/returns at the dealership and p1 is the probability of
        this number occuring.
        """
        d = defaultdict(float)
        for demand, p_demand in tpoiss2(lamb=demand_rate, k_max=cars): # demand is the demand (number of cars), p_demand is the probability of this demand.
            rented = min(cars, demand) # you cannot rent more cars than there are at the dealership regardless of demand.
            d_cars = cars - rented # new cars after rental. Now time for returns
            total_reward = rented * self.car_rent # car rentals give us profit!
            for returns, p_returns in tpoiss2(lamb=return_rate, k_max=self.max_cars - d_cars):
                new_cars = min(d_cars + returns, self.max_cars)
                if new_cars < 0:
                    raise Exception("Bad car number", new_cars)
                d[(new_cars, total_reward)] += p_demand * p_returns
        return d

    def Psr(self, s, a):
        # TODO: 1 lines missing.
        raise NotImplementedError("Update cars moved s")
        if not all( x >= 0 for x in s ):
            raise Exception("Bad state", s)

        """ 
        This is just a possible way to solve the problem, but I did the following:
        
        For each dealership 1 and 2, the two dictionaries below contains the following:
        Keys: values of the form (new_cars, reward): probability of the (possible) new cars at the dealership and the probability of this 
        particular outcome occuring. You can use this to compute the full transition probability.
        """
        dS1 = self.overnight_in_out(s[0], demand_rate=self.demand_rate_1, return_rate=self.return_rate_1)
        dS2 = self.overnight_in_out(s[1], demand_rate=self.demand_rate_2, return_rate=self.return_rate_2)

        """ 
        Remember to return a dictionary d of the form:
        d[ ( (cars_dealership_1, cars_dealership_2), reward)] = probability
        
        where the cars at each dealership are contained in dS1, dS2, the reward is the sum of rewards in dS1, dS2 PLUS the cost of moving the cars: |a|*move_cost
        and the probabilities are also computed using the probabilities in dS1, dS2. 
        As a check, you can see if d.values() sum to 1. 
        """
        d = defaultdict(float)
        for (c1, reward_1), pc1 in dS1.items():
            for (c2, reward_2), pc2 in dS2.items():
                # TODO: 1 lines missing.
                raise NotImplementedError("Use dS1, dS2 to compute new state, reward transition probabilities (see MDP class)")
        return d

class CacheMDP(MDP):
    def __init__(self, mdp, compress_tol=0.):
        """ A small MDP transformer class which help speed up computations by caching intermediate results and
        discarding very low-probability transitions. """
        self.verbose = True
        timer = Timer()
        timer.tic("states")
        print("Setting up cache for MDP. Constructing set of states")
        len(mdp.states)
        print("Total state count", len(mdp.states))
        timer.toc()
        print(timer.display())
        self._Psr_cache = [ [None]*len(mdp.A(s)) for s in mdp.states]
        self._Psr_cache = {}
        import sys
        from tqdm import tqdm
        print("Iterating over all transition probabilities...")
        ntot = 0
        with tqdm(total=len(mdp.states), disable=not self.verbose, file=sys.stdout) as tq:
            for s in mdp.nonterminal_states:
                for a in mdp.A(s):
                    Psr_ = mdp.Psr(s, a)
                    pv = list(Psr_.values())
                    pmax = max(pv)
                    prem = sum([p for p in pv if p >= pmax * compress_tol]) # For normalizing probabilities.
                    l = []
                    for (sp, r), p in Psr_.items():
                        if p > pmax * compress_tol:
                            l.append( (sp, r, p/prem))
                    ntot += len(l)
                    self._Psr_cache[(s,a)] = l
                tq.set_postfix({'total cache size': ntot})
                tq.update()

        self.mdp = mdp
        self._nonterminal_states = mdp._nonterminal_states # [s for s in self.states if not self.is_terminal(s)]
        self._states = mdp._states
        self.A = mdp.A
        self.is_terminal = mdp.is_terminal

    def Psr(self, state, action):
        l = self._Psr_cache[(state,action)]
        d = { (sp, r): p for sp, r, p in l }
        return d


def jack_experiment(max_cars=4, max_iters=10**6, use_cache=True, compress_tol=0.01):
    def plt_jack(fun, max_cars, name, pdf):
        import seaborn as sns
        figsize = 8
        plt.figure(figsize=(figsize, figsize))
        A = np.zeros((max_cars+1, max_cars+1))
        for (i,j), v in fun.items():
            A[i,j] = v
        sns.heatmap(A, cmap="YlGnBu", annot=True, cbar=False, square=True, fmt='g')

        # plot_value_function(env, A)
        plt.ylabel("Cars at dealership 1")
        plt.xlabel("Cars at dealership 2")
        plt.title(f"{name} for Jacks car rental ({max_cars} cars)")
        plt.gca().invert_yaxis()
        savepdf(pdf)
        plt.show()

    timer = Timer(start=True)
    timer.tic("small")
    env = JackRentalMDP(max_cars=max_cars, verbose=True)
    if use_cache:
        print("building cache")
        env = CacheMDP(env, compress_tol=compress_tol)
    timer.toc()
    timer.tic("small VI")
    print("value iteration...")
    pi, V = value_iteration(env, gamma=.9, theta=1e-1, max_iters=max_iters, verbose=True)

    timer.toc()
    print("", len(env.states))
    print(timer.display())

    plt_jack(V, max_cars, name="Value function", pdf=f"jacks_{max_cars}_value")
    plt_jack(pi, max_cars, name="Optimal policy", pdf=f"jacks_{max_cars}_policy")

if __name__ == "__main__":
    """
    Jack's car rental problem (SB18, Example 4.2)
    """
    print("Solving the small problem")
    jack_experiment(max_cars=4, use_cache=False)

    print("Solving the large problem (warning, will take a few minutes)")
    jack_experiment(max_cars=20, use_cache=True, max_iters=100, compress_tol=0.001)
