"""
This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.

References:
  [Her21] Tue Herlau. Sequential decision making. (See 02465_Notes.pdf), 2021.

"""
import numpy as np
import sympy as sym
import sys
from scipy.optimize import Bounds, minimize
from scipy.interpolate import interp1d
from irlc.ex04.continuous_time_model import symv
from irlc.ex04.continuous_time_discretized_model import sympy_modules_
from irlc import Timer
from tqdm import tqdm

def bounds2fun(t0, tF, bounds):
    """
    Given start and end times [t0, tF] and a bounds object with upper/lower bounds on some variable x:

    bounds.lb <= x <= bounds.ub

    this function returns a new function f such that f(t0) equals bounds.lb and f(tF) = bounds.ub and
    f(t) interpolates between the uppower/lower bounds linearly.
    """
    return interp1d(np.asarray([t0, tF]), np.stack([np.reshape(b, (-1,)) for b in bounds], axis=1))

def direct_solver(env, options):
    """
    Main direct solver method, see (Her21, Algorithm 21). Given a list of options of length S, the solver performers collocation
    using the options[i], and use the result of options[i] to initialize collocation on options[i+1].
    This iterative refinement is required to obtain good overall solutions.
    """
    if isinstance(options, dict):
        options = [options]
    solutions = []  # re-use result of current solutions to initialize next with higher value of N
    for i, opt in enumerate(options):
        optimizer_options = opt['optimizer_options']  # to be passed along to minimize()
        if i == 0 or "guess" in opt:
            # No guess functions are given. Re-calculate by linearly interpreting bounds (see (Her21, Subsection 10.3.4))
            guess = opt['guess']
            guess['u'] = bounds2fun(guess['t0'],guess['tF'],guess['u']) if isinstance(guess['u'], list) else guess['u']
            guess['x'] = bounds2fun(guess['t0'],guess['tF'],guess['x']) if isinstance(guess['x'], list) else guess['x']
        else:
            # For an iterative solver ((Her21, Subsection 10.3.4)), initialize the guess at iteration i to be the solution at iteration i-1.
            # The guess consists of a guess for t0, tF (just numbers) as well as x, u (state/action trajectories),
            # the later two being functions. The format of the guess is just a dictionary (you have seen several examples)
            # i.e. guess = {'t0': (number), 'tF': (number), 'x': (function), 'u': (function)}
            # and you can get the solution by using solutions[i - 1]['fun']. (insert a breakpoint and check the fields)
            # TODO: 1 lines missing.
            raise NotImplementedError("Define guess = {'t0': ..., ...} here.")
        N = opt['N']
        print(f"{i}> Collocation restart N={N}")
        sol = collocate(env, N=N, optimizer_options=optimizer_options, guess=guess, verbose=opt.get('verbose', False))
        solutions.append(sol)

    print("Collocation success?")
    for i, s in enumerate(solutions):
        print(f"{i}> Success? {s['solver']['success']}")
    return solutions

def collocate(env, N=25, optimizer_options=None, guess=None, verbose=True):
    """ Create all symbolic variables that will be used in the remainder. """
    timer = Timer(start=True)
    t0, tF = sym.symbols("t0"), sym.symbols("tF")
    ts = t0 + np.linspace(0, 1, N) * (tF-t0)   # N points linearly spaced between [t0, tF]
    xs, us = [], []
    for i in range(N):
        xs.append(list(symv("x_%i_" % i, env.state_size)))
        us.append(list(symv("u_%i_" % i, env.action_size)))

    ''' Construct guess z0, all simple bounds [z_lb, z_ub] for the problem and collect all symbolic variables as z '''
    sb = env.simple_bounds()  # get simple inequality boundaries in problem (v_lb <= v <= v_ub)
    z = []  # list of all *symbolic* variables in the problem
    z0, z_lb, z_ub = [], [], []  # Guess z0 and lower/upper bounds (list-of-numbers): z_lb[k] <= z0[k] <= z_ub[k]
    ts_eval = sym.lambdify((t0, tF), ts, modules='numpy')
    for k in range(N):
        x_bnd, u_bnd = sb['x'], sb['u']
        if k == 0:
            x_bnd = sb['x0']
        if k == N - 1:
            x_bnd = sb['xF']
        tk = ts_eval(guess['t0'], guess['tF'])[k]
        """ In these lines, update z, z0, z_lb, and z_ub with values corresponding to xs[k], us[k]. 
        The values are all lists; i.e. z[j] (symbolic) has guess z0[j] (float) and bounds z_lb[j], z_ub[j] (floats) """
        # TODO: 2 lines missing.
        raise NotImplementedError("Updates for x_k, u_k")

    """ Update z, z0, z_lb, and z_ub with bounds/guesses corresponding to t0 and tF (same format as above). """
    # TODO: 2 lines missing.
    raise NotImplementedError("Updates for t0, tF")
    if verbose:
        print(f"z={z}\nz0={np.asarray(z0).round(1).tolist()}\nz_lb={np.asarray(z_lb).round(1).tolist()}\nz_ub={np.asarray(z_ub).round(1).tolist()}") 
    print(">>> Trapezoid collocation of problem") # problem in this section
    fs, cs = [], []  # lists of symbolic variables corresponding to f_k and c_k, see (Her21, Algorithm 20).
    for k in range(N):
        """ Update both fs and cs; these are lists of symbolic expressions such that fs[k] corresponds to f_k and cs[k] to c_k in the slides. 
        Use the functions env.sym_f and env.sym_c """
        # fs.append( symbolic variable corresponding to f_k; see env.sym_f). similarly update cs.append(env.sym_c(...) ).
        # TODO: 2 lines missing.
        raise NotImplementedError("Compute f[k] and c[k] here (see slides) and add them to above lists")

    J = env.sym_cf(x0=xs[0], t0=t0, xF=xs[-1], tF=tF)  # cost, to get you started, but needs more work
    eqC, ineqC = [], []  # all symbolic equality/inequality constraints are stored in these lists
    for k in range(N - 1):
        # Update cost function ((Her21, eq. (10.15))). Use the above defined symbolic expressions ts, hk and cs.
        # TODO: 2 lines missing.
        raise NotImplementedError("Update J here")
        # Set up equality constraints. See (Her21, eq. (10.18)).
        for j in range(env.state_size):
            """ Create all collocation equality-constraints here and add them to eqC. I.e.  
            xs[k+1] - xs[k] = 0.5 h_k (f_{k+1} + f_k)
            Note we have to create these coordinate-wise which is why we loop over j. 
            """
            # TODO: 1 lines missing.
            raise NotImplementedError("Update collocation constraints here")
        """
        To solve problems with dynamical path constriants like Brachiostone, update ineqC here to contain the 
        inequality constraint env.sym_h(...) <= 0. """
        # TODO: 1 lines missing.
        raise NotImplementedError("Update symbolic path-dependent constraint h(x,u,t)<=0 here")

    print(">>> Creating objective and derivative...")
    timer.tic("Building symbolic objective")
    J_fun = sym.lambdify([z], J, modules='numpy')  # create a python function from symbolic expression
    # To compute the Jacobian, you can use sym.derive_by_array(J, z) to get the correct symbolic expression, then use sym.lamdify (as above) to get a numpy function.
    # TODO: 1 lines missing.
    raise NotImplementedError("Jacobian of J. See how this is computed for equality/inequality constratins for help.")
    if verbose:
        print(f"eqC={eqC}\nineqC={ineqC}\nJ={J}") 
    timer.toc()
    print(">>> Differentiating equality constraints..."), timer.tic("Differentiating equality constraints")
    constraints = []
    for eq in tqdm(eqC, file=sys.stdout):  # dont' write to error output.
        constraints.append(constraint2dict(eq, z, type='eq'))
    timer.toc()
    print(">>> Differentiating inequality constraints"), timer.tic("Differentiating inequality constraints")
    constraints += [constraint2dict(ineq, z, type='ineq') for ineq in ineqC]
    timer.toc()

    c_viol = sum(abs(np.minimum(z_ub - np.asarray(z0), 0))) + sum(abs(np.maximum(np.asarray(z_lb) - np.asarray(z0), 0)))
    if c_viol > 0:  # check if: z_lb <= z0 <= z_ub. Violations only serious if large
        print(f">>> Warning! Constraint violations found of total magnitude: {c_viol:4} before optimization")

    print(">>> Running optimizer..."), timer.tic("Optimizing")
    z_B = Bounds(z_lb, z_ub)
    res = minimize(J_fun, x0=z0, method='SLSQP', jac=J_jac, constraints=constraints, options=optimizer_options, bounds=z_B) 
    # Compute value of equality constraints to check violations
    timer.toc()
    eqC_fun = sym.lambdify([z], eqC)
    eqC_val_ = eqC_fun(res.x)
    eqC_val = np.zeros((N - 1, env.state_size))

    x_res = np.zeros((N, env.state_size))
    u_res = np.zeros((N, env.action_size))
    t0_res = res.x[-2]
    tF_res = res.x[-1]

    m = env.state_size + env.action_size
    for k in range(N):
        dx = res.x[k * m:(k + 1) * m]
        if k < N - 1:
            eqC_val[k, :] = eqC_val_[k * env.state_size:(k + 1) * env.state_size]
        x_res[k, :] = dx[:env.state_size]
        u_res[k, :] = dx[env.state_size:]

    # Generate solution structure
    ts_numpy = ts_eval(t0_res, tF_res)
    # make linear interpolant similar to (Her21, eq. (10.22))
    ufun = interp1d(ts_numpy, np.transpose(u_res), kind='linear')
    # Evaluate knot points (useful for debugging but not much else):
    f_eval = sym.lambdify((t0, tF, xs, us), fs)
    fs_numpy = f_eval(t0_res, tF_res, x_res, u_res)
    fs_numpy = np.asarray(fs_numpy)

    """ make cubic interpolant similar to (Her21, eq. (10.26)) """
    x_fun = lambda t_new: trapezoid_interpolant(ts_numpy, np.transpose(x_res), np.transpose(fs_numpy), t_new=t_new)

    if verbose:
        newt = np.linspace(ts_numpy[0], ts_numpy[-1], len(ts_numpy)-1)
        print( x_fun(newt) ) 

    sol = {
        'grid': {'x': x_res, 'u': u_res, 'ts': ts_numpy, 'fs': fs_numpy},
        'fun': {'x': x_fun, 'u': ufun, 'tF': tF_res, 't0': t0_res},
        'solver': res,
        'eqC_val': eqC_val,
        'inputs': {'z': z, 'z0': z0, 'z_lb': z_lb, 'z_ub': z_ub},
    }
    print(timer.display())
    return sol

def trapezoid_interpolant(ts, xs, fs, t_new=None):
    ''' Quadratic interpolant as in (Her21, eq. (10.26)). Inefficient but works. '''
    I = []
    t_new = np.reshape(np.asarray(t_new), (-1,))
    for t in t_new:  # yah, this is pretty terrible..
        i = -1
        for i in range(len(ts) - 1):
            if ts[i] <= t and t <= ts[i + 1]:
                break
        I.append(i)

    ts = np.asarray(ts)
    I = np.asarray(I)
    tau = t_new - ts[I]
    hk = ts[I + 1] - ts[I]
    """
    Make interpolation here. Should be a numpy array of dimensions [xs.shape[0], len(I)]
    What the code does is that for each t in ts, we work out which knot-point interval the code falls within. I.e. 
    insert a breakpoint and make sure you understand what e.g. the code tau = t_new - ts[I] does.
     
    Given this information, we can recover the relevant (evaluated) knot-points as for instance 
    fs[:,I] and those at the next time step as fs[:,I]. With this information, the problem is simply an 
    implementation of  (Her21, eq. (10.26)), i.e. 

    > x_interp = xs[:,I] + tau * fs[:,I] + (...)    
    
    """
    # TODO: 1 lines missing.
    raise NotImplementedError("")
    return x_interp


def constraint2dict(symb, all_vars, type='eq'):
    ''' Turn constraints into a dict with type, fun, and jacobian field. '''
    if type == "ineq": symb = -1 * symb  # To agree with sign convention in optimizer

    f = sym.lambdify([all_vars], symb, modules=sympy_modules_)
    # np.atan = np.arctan  # Monkeypatch numpy to contain atan. Passing "numpy" does not seem to fix this.
    jac = sym.lambdify([all_vars], sym.derive_by_array(symb, all_vars), modules=sympy_modules_)
    eq_cons = {'type': type,
               'fun': f,
               'jac': jac}
    return eq_cons

def get_opts(N, ftol=1e-6, guess=None, verbose=False): # helper function to instantiate options objet.
    d = {'N': N,
         'optimizer_options': {'maxiter': 1000,
                               'ftol': ftol,
                               'iprint': 1,
                               'disp': True,
                               'eps': 1.5e-8},  # 'eps': 1.4901161193847656e-08,
         'verbose': verbose}
    if guess:
        d['guess'] = guess
    return d

def run_direct_small_problem():
    from irlc.ex04.model_pendulum import ContiniousPendulumModel
    env = ContiniousPendulumModel()
    """
    Test out implementation on a VERY small grid. This will work fairly terribly, but we can print out the various symbolic expressions
    using verbose=True
    """
    print("Solving with a small grid, N=5 (yikes)")
    options = [get_opts(N=5, ftol=1e-3, guess=env.guess(), verbose=True)]
    solutions = direct_solver(env, options)
    return env, solutions

if __name__ == "__main__":
    from irlc.ex05.direct_plot import plot_solutions
    env, solutions = run_direct_small_problem()
    plot_solutions(env, solutions, animate=False, pdf="direct_pendulum_small")
