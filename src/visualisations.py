import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.ticker import MaxNLocator

def plot_forecast(data_list, assets, colours):

    num_plots = len(data_list)

    gridsize = (num_plots, 4)
    fig = plt.figure(figsize=(14, 3 * num_plots))
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle('Volatility Forecast with Stochastic Volatility', fontsize=16)
    
    # Define percentiles for the 5 layers
    percentiles = [(1, 99), (5, 95), (15, 85),  (30, 60)]
    alphas = [0.2, 0.3, 0.4, 0.6]

    for i in range(num_plots):
        # Identifying variables:
        hist_price = data_list[i][0]
        sim_price = data_list[i][1]
        asset = assets[i]
        colour = colours[i]
        # Setting graph up
        axl = plt.subplot2grid(gridsize, (i, 0), colspan=3, rowspan=1)
        axr = plt.subplot2grid(gridsize, (i, 3))

        # plotting simulations
        axl.plot(hist_price.index[:-6], hist_price.values[:-6], label=asset, color=colour, linewidth=2 )
        axl.plot(hist_price.index[-7:], hist_price.values[-7:], color=colour, linestyle='--')

        # Calculate and plot the layers
        for (p_low, p_high), alpha in zip(percentiles, alphas):
            lower_bound = np.percentile(sim_price, p_low, axis=0)
            upper_bound = np.percentile(sim_price, p_high, axis=0)
            axl.fill_between(hist_price.index[-7:], lower_bound, upper_bound, color=colour, alpha=alpha)

        # Plotting histogram
        axr.hist(sim_price[:,-1], density=True, bins=30, orientation="horizontal",edgecolor=colour, color = colour, alpha=0.6)

        # Plotting KDE
        kde = gaussian_kde(sim_price[:,-1])
        x =  np.linspace(axl.get_ylim()[0], axl.get_ylim()[1], 1000)
        axr.plot(kde(x), x, color=colour )

        # Formatting plots
        y_lim = (min(axl.get_ylim()[0], axr.get_ylim()[0]), max(axl.get_ylim()[1], axr.get_ylim()[1]))
        axl.set_xlabel("Date")
        axl.set_ylabel("Price")
        axl.set_ylim(y_lim)
        axl.xaxis.set_major_locator(MaxNLocator(nbins=7))
        axl.yaxis.set_major_locator(MaxNLocator(nbins=7))
        # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: pd.Timestamp(x).strftime('%Y-%m-%d')))


        axr.set_ylim(y_lim) 
        axr.set_xlabel("Probability")
        axl.legend(loc=2)
        
        axl.grid(True)

    return fig