from irlc import main_plot
import matplotlib.pyplot as plt
gammas = [0.95,0.99,1]
grid_sizes = [[10,10],[15,15],[20,20]]

for grid_size in grid_sizes:
    exp_names = []
    for gamma in gammas:
        exp_names.append(f"experiments/grid{grid_size[0]}x{grid_size[0]}/q_gamma{gamma}")
    main_plot(exp_names, smoothing_window=10, resample_ticks=10)
    plt.ylim([-150, 3000])
    plt.title(f"Q-learning on snake - {grid_size[0]}x{grid_size[0]} grid")
    plt.show()