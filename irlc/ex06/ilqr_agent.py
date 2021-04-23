"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

"""
from irlc.ex06.model_rendevouz import RendevouzEnvironment
from irlc.ex06.ilqr_rendovouz_basic import ilqr
from irlc import train
from irlc import Agent

class ILQRAgent(Agent):
    def __init__(self, env, discrete_model, N=250, ilqr_iterations=10, use_ubar=False, use_linesearch=True):
        super().__init__(env)
        self.dt = discrete_model.dt
        x0 = discrete_model.reset()
        xs, us, self.J_hist, L, l = ilqr(discrete_model, N, x0, n_iter=ilqr_iterations, use_linesearch=use_linesearch)
        self.ubar = us
        self.xbar = xs
        self.L = L
        self.l = l
        self.use_ubar = use_ubar # Should policy use open-loop u-bar (suboptimal) or closed-loop L_k, l_k?

    def pi(self, x, t=None):
        k = int(t / self.dt)
        if self.use_ubar:
            u = self.ubar[k]
        else:
            if k >= len(self.ubar):
                print(k, len(self.ubar))
                k = len(self.ubar)-1
            # See (Her21, eq. (12.17))
            # TODO: 1 lines missing.
            raise NotImplementedError("Generate action using the control matrices.")
        return u

def solve_rendevouz():
    env = RendevouzEnvironment()
    N = int(env.Tmax / env.dt)
    agent = ILQRAgent(env, env.discrete_model, N=N)
    stats, trajectories = train(env, agent, num_episodes=1, return_trajectory=True)
    env.close()
    return stats, trajectories, agent

if __name__ == "__main__":
    from irlc.ex06.ilqr_rendovouz_basic import plot_vehicles
    import matplotlib.pyplot as plt
    stats, trajectories, agent = solve_rendevouz()
    t =trajectories[0].state
    xb = agent.xbar
    plot_vehicles(t[:,0], t[:,1], t[:,2], t[:,3], linespec=':', legend=("RK4 policy simulation", "RK4 policy simulation"))
    plot_vehicles(xb[:,0], xb[:,1], xb[:,2], xb[:,3], linespec='-')
    plt.legend()
    plt.show()
