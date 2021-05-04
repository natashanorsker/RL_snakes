from irlc import main_plot
import matplotlib.pyplot as plt
gammas = [0.9,0.95,1]
grid_sizes = [[10,10],[15,15],[20,20]]
betas = [0.1,0.2]
for grid_size in grid_sizes:
    exp_names = []
    for gamma in gammas:
        exp_names.append(f"experiments/grid{grid_size[0]}x{grid_size[0]}/q_gamma{gamma}")
    for beta in betas:
        exp_names.append(f"experiments/grid{grid_size[0]}x{grid_size[0]}/r_beta{beta}")
    main_plot(exp_names, smoothing_window=100, resample_ticks=250)
    # plt.ylim([-150, 3000])
    plt.title(f"Q/R-learning on Snake - {grid_size[0]}x{grid_size[0]} grid")
    plt.show()