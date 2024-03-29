import matplotlib.pyplot as plt
import numpy as np


def plot_results(measures, target, estimates, dates, title, path):
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.figure()
    plt.plot(measures, 'k+', label='noisy measurements')
    plt.plot(estimates, 'b-', label='prediction')
    plt.plot(target, color='g', label='true value')
    plt.title(title, fontweight='bold')

    plt.xlabel('dates')
    x_label_count = int(len(dates) / 70)
    xi = list(range(x_label_count))
    xi = [i * 70 for i in xi]
    x_axes = [dates[i] for i in xi]
    plt.xticks(xi, x_axes)

    plt.ylabel('volume per hour')
    plt.legend()
    plt.savefig(path)
    plt.show()

    """ peak times: [07:30- 08:30], [16:30-17:30] """
    """ peak ticks = [90, 102], [198, 210] """
    """ ticks range 0-287 """
    """ midnight tick => |0-90| = 90"""


def get_trend(point):
    array = np.asarray([89, 101, 197, 209])
    dist = np.min([np.abs(array - point), np.abs(array - (point + 288))])
    if 87 < point < 103 or 195 < point < 211:
        dist = 2

    trend_val = np.log(dist)

    return 2 / trend_val
