from irlc.utils.irlc_plot import get_datasets, plot_data
import matplotlib.pyplot as plt

def plot_lengths(experiments, legends=None, smoothing_window=None, resample_ticks=None,
              x_key="Episode",
              y_key='Accumulated Reward',
              no_shading=False,
              **kwargs):
    if no_shading:
        kwargs['units'] = 'Unit'
        kwargs['estimator'] = None

    ensure_list = lambda x: x if isinstance(x, list) else [x]
    experiments = ensure_list(experiments)

    if legends is None:
        legends = experiments
    legends = ensure_list(legends)

    data = []
    for logdir, legend_title in zip(experiments, legends):
        resample_key = x_key if resample_ticks is not None else None
        data += get_datasets(logdir, x=x_key, condition=legend_title, smoothing_window=smoothing_window, resample_key=resample_key, resample_ticks=resample_ticks)

    y_key = "Max Snake Length"

    for i, df in enumerate(data):
        data[i]["Accumulated Reward"] = ((df["Accumulated Reward"] + df["Length"] - 3)/101 + 4).cummax()
        data[i] = data[i].rename(columns={"Accumulated Reward": y_key})
        #print(data[i]["Max Snake Length"])

    plot_data(data, y=y_key, x=x_key, **kwargs)


if __name__ == "__main__":
    gammas = [0.9]
    grid_sizes = [[15, 15]]
    betas = [0.2]
    for grid_size in grid_sizes:
        exp_names = []
        for gamma in gammas:
            exp_names.append(f"experiments/grid{grid_size[0]}x{grid_size[0]}/q_gamma{gamma}")
        for beta in betas:
            exp_names.append(f"experiments/grid{grid_size[0]}x{grid_size[0]}/r_beta{beta}")
        plot_lengths(exp_names, resample_ticks=None)
        plt.title(f"Q/R-learning on Snake - {grid_size[0]}x{grid_size[0]} grid")
        plt.show()